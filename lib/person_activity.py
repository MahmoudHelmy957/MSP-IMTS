import os
import lib.utils as utils
import numpy as np
import tarfile
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from lib.utils import get_device

# Adapted from: https://github.com/rtqichen/time-series-datasets

class PersonActivity(object):
	urls = [
		'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
	]

	tag_ids = [
		"010-000-024-033", #"ANKLE_LEFT",
		"010-000-030-096", #"ANKLE_RIGHT",
		"020-000-033-111", #"CHEST",
		"020-000-032-221" #"BELT"
	]
	
	tag_dict = {k: i for i, k in enumerate(tag_ids)}

	label_names = [
		 "walking",
		 "falling",
		 "lying down",
		 "lying",
		 "sitting down",
		 "sitting",
		 "standing up from lying",
		 "on all fours",
		 "sitting on the ground",
		 "standing up from sitting",
		 "standing up from sit on grnd"
	]

	#Merge similar labels into one class
	label_dict = {
		"walking": 0,
		 "falling": 1,
		 "lying": 2,
		 "lying down": 2,
		 "sitting": 3,
		 "sitting down" : 3,
		 "standing up from lying": 4,
		 "standing up from sitting": 4,
		 "standing up from sit on grnd": 4,
		 "on all fours": 5,
		 "sitting on the ground": 6
		 }

	def __init__(self, root, download=True,
		reduce='average', max_seq_length = None,
		n_samples = None, device = torch.device("cpu")):

		self.root = root
		self.reduce = reduce
		# self.max_seq_length = max_seq_length

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

		if n_samples is not None:
			print('Total records:', len(self.data))
			self.data = self.data[:n_samples]

	def download(self):
		if self._check_exists():
			return

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		os.makedirs(self.raw_folder, exist_ok=True)
		os.makedirs(self.processed_folder, exist_ok=True)

		def save_record(records, record_id, tt, vals, mask, labels):
			tt = torch.tensor(tt).to(self.device)

			vals = torch.stack(vals)
			mask = torch.stack(mask)
			labels = torch.stack(labels)

			# flatten the measurements for different tags
			vals = vals.reshape(vals.size(0), -1)
			mask = mask.reshape(mask.size(0), -1)
			assert(len(tt) == vals.size(0))
			assert(mask.size(0) == vals.size(0))
			assert(labels.size(0) == vals.size(0))

			records.append((record_id, tt, vals, mask))


		for url in self.urls:
			filename = url.rpartition('/')[2]
			download_url(url, self.raw_folder, filename, None)

			print('Processing {}...'.format(filename))

			dirname = os.path.join(self.raw_folder)
			records = []
			first_tp = None

			for txtfile in os.listdir(dirname):
				with open(os.path.join(dirname, txtfile)) as f:
					lines = f.readlines()
					prev_time = -1
					tt = []

					record_id = None
					for l in lines:
						cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
						value_vec = torch.Tensor((float(val1), float(val2), float(val3))).to(self.device)
						time = float(time)

						if cur_record_id != record_id:
							if record_id is not None:
								save_record(records, record_id, tt, vals, mask, labels)
							tt, vals, mask, nobs, labels = [], [], [], [], []
							record_id = cur_record_id
						
							tt = [torch.zeros(1).to(self.device)]
							vals = [torch.zeros(len(self.tag_ids),3).to(self.device)]
							mask = [torch.zeros(len(self.tag_ids),3).to(self.device)]
							nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
							labels = [torch.zeros(len(self.label_names)).to(self.device)]
							
							first_tp = time
							time = round((time - first_tp)/ 10**4)
							prev_time = time
						else:
							time = round((time - first_tp)/ 10**4) # quatizing by 1000 ms. 10,000 is one millisecond, 10,000,000 is one second

						if time != prev_time:
							tt.append(time)
							vals.append(torch.zeros(len(self.tag_ids),3).to(self.device))
							mask.append(torch.zeros(len(self.tag_ids),3).to(self.device))
							nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
							labels.append(torch.zeros(len(self.label_names)).to(self.device))
							prev_time = time

						if tag_id in self.tag_ids:
							n_observations = nobs[-1][self.tag_dict[tag_id]]
							if (self.reduce == 'average') and (n_observations > 0):
								prev_val = vals[-1][self.tag_dict[tag_id]]
								new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
								vals[-1][self.tag_dict[tag_id]] = new_val
							else:
								vals[-1][self.tag_dict[tag_id]] = value_vec

							mask[-1][self.tag_dict[tag_id]] = 1
							nobs[-1][self.tag_dict[tag_id]] += 1

							if label in self.label_names:
								if torch.sum(labels[-1][self.label_dict[label]]) == 0:
									labels[-1][self.label_dict[label]] = 1
						else:
							assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
					save_record(records, record_id, tt, vals, mask, labels)
			
			print('# of records after processed:', len(records))
			torch.save(
				records,
				os.path.join(self.processed_folder, 'data.pt')
			)
				
		print('Done!')

	def _check_exists(self):
		for url in self.urls:
			filename = url.rpartition('/')[2]
			if not os.path.exists(
				os.path.join(self.processed_folder, 'data.pt')
			):
				return False
		return True

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def data_file(self):
		return 'data.pt'

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Max length: {}\n'.format(self.max_seq_length)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str

def get_person_id(record_id):
	# The first letter is the person id
	person_id = record_id[0]
	person_id = ord(person_id) - ord("A")
	return person_id


def Activity_time_chunk(data, args, device):

	chunk_data = []
	history = args.history # ms
	pred_window = args.pred_window # ms
	for b, (record_id, tt, vals, mask) in enumerate(data):
		t_max = int(tt.max())
		for st in range(0, t_max - history, pred_window):
			et = st + history + pred_window
			if(et >= t_max):
				idx = torch.where((tt >= st) & (tt <= et))[0]
			else:
				idx = torch.where((tt >= st) & (tt < et))[0]
			new_id = f"{record_id}_{st//pred_window}"
			chunk_data.append((new_id, tt[idx] - st, vals[idx], mask[idx]))

	return chunk_data

def Activity_get_seq_length(args, records):
	
	max_input_len = 0
	max_pred_len = 0
	lens = []
	for b, (record_id, tt, vals, mask) in enumerate(records):
		n_observed_tp = torch.lt(tt, args.history).sum()
		max_input_len = max(max_input_len, n_observed_tp)
		max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
		lens.append(n_observed_tp)
	lens = torch.stack(lens, dim=0)
	median_len = lens.median()

	return max_input_len, max_pred_len, median_len


##############################################################
# Multi-scale collate for PersonActivity
##############################################################
def patch_variable_time_collate_fn_ms_activity(
    batch, args, device=torch.device("cpu"),
    data_min=None, data_max=None, time_max=None,
    scales_ms=(100.0, 300.0), strides_ms=None, history_ms=3000.0
):
    """
    Same as physionet.patch_variable_time_collate_fn_ms but using ms units.
    Returns:
      X_list, tt_list, mk_list, npatches,
      tp_to_predict, data_to_predict, mask_predicted_data
    """
    if strides_ms is None:
        strides_ms = scales_ms

    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True
    )

    n_observed_tp = torch.lt(combined_tt, args.history).sum()
    observed_tt = combined_tt[:n_observed_tp]

    B = len(batch)
    combined_vals = torch.zeros([B, len(combined_tt), D], device=device)
    combined_mask = torch.zeros_like(combined_vals)

    predicted_tp_list, predicted_data_list, predicted_mask_list = [], [], []
    offset = 0
    for b, (record_id, tt, vals, mask) in enumerate(batch):
        idx = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)
        combined_vals[b, idx] = vals.to(device)
        combined_mask[b, idx] = mask.to(device)

        n_obs_cur = torch.lt(tt, args.history).sum()
        predicted_tp_list.append(tt[n_obs_cur:])
        predicted_data_list.append(vals[n_obs_cur:])
        predicted_mask_list.append(mask[n_obs_cur:])

    combined_vals = combined_vals[:, :n_observed_tp]
    combined_mask = combined_mask[:, :n_observed_tp]

    from torch.nn.utils.rnn import pad_sequence
    predicted_tp   = pad_sequence(predicted_tp_list,   batch_first=True)
    predicted_data = pad_sequence(predicted_data_list, batch_first=True)
    predicted_mask = pad_sequence(predicted_mask_list, batch_first=True)

    if args.dataset != 'ushcn':
        combined_vals = utils.normalize_masked_data(combined_vals, combined_mask, att_min=data_min, att_max=data_max)
        predicted_data = utils.normalize_masked_data(predicted_data, predicted_mask, att_min=data_min, att_max=data_max)

    observed_tt  = utils.normalize_masked_tp(observed_tt,  att_min=0, att_max=time_max)
    predicted_tp = utils.normalize_masked_tp(predicted_tp, att_min=0, att_max=time_max)

    single = {
        "data": combined_vals,
        "time_steps": observed_tt,
        "mask": combined_mask,
        "data_to_predict": predicted_data,
        "tp_to_predict": predicted_tp,
        "mask_predicted_data": predicted_mask,
    }

    ms = utils.multiscale_split_and_patch_batch(
        data_dict=single, args=args, history_hours=float(history_ms),
        scales_hours=list(scales_ms), strides_hours=list(strides_ms)
    )

    return {
        "X_list": ms["X_list"], "tt_list": ms["tt_list"], "mk_list": ms["mk_list"],
        "npatches": ms["npatches"],
        "tp_to_predict": single["tp_to_predict"],
        "data_to_predict": single["data_to_predict"],
        "mask_predicted_data": single["mask_predicted_data"],
    }



if __name__ == '__main__':
	torch.manual_seed(1991)

	dataset = PersonActivity('data/PersonActivity', download=True)
	dataloader = DataLoader(dataset, batch_size=30, shuffle=True, collate_fn= variable_time_collate_fn_activity)
	dataloader.__iter__().next()
