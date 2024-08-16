import random
import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset


def history_process(history, news, args):
    content = []
    history = history.split()[:args.history_limit]  # limit the history length
    for item in history:
        content.append(prompt_gen(news, item))
    return content


def candidate_process(candidate, pd_news, args):
    pos_news = [impression.split('-')[0] for impression in candidate.split() if impression.endswith('-1')]
    neg_news = [impression.split('-')[0] for impression in candidate.split() if impression.endswith('-0')]
    # filter out samples without positive sample or negativa sample
    if not pos_news or not neg_news:
        return None
    for news in pos_news:
        candidate = []
        # num_neg_samples <= k
        num_neg_samples = min(len(neg_news), args.num_neg)
        sampled_negatives = random.sample(neg_news, num_neg_samples)
        # import pdb; pdb.set_trace()
        candidate.append(news)
        candidate.extend(sampled_negatives)  # concat postive news and negative news
    content = []
    for item in candidate:
        content.append(prompt_gen(pd_news, item))
    return content


# transform the news ID into words
def prompt_gen(news, item):
    # news: pandas frame from news.tsv
    category, sub_category, title, abstract = news.loc[item][1], news.loc[item][2], news.loc[item][3], news.loc[item][4]
    prompt = "This is a news, whose category is <<<{}>>>, and the subcategory is <<<{}>>>. The title is <<<{}>>>, and the abstract is <<<{}>>>.".format(category, sub_category, title, abstract)

    return prompt


class UserDataset(Dataset):
	"""
	getitem return history's click and new to judge.
	"""
	def __init__(self, file):
		self.all = pd.read_csv(file, delimiter='\t',header=None)
		self.all = self.all.drop(self.all[pd.isna(self.all[3])].index)
	def __len__(self):
		return self.all.shape[0]
	def __getitem__(self, idx):
		return self.all.iloc[idx,3], self.all.iloc[idx,4]



class TestDataset(Dataset):
    def __init__(self, user_data, item_embeddings, get_embedding, device):
        self.user_data = user_data
        self.item_embeddings = item_embeddings
        self.get_embedding = get_embedding
        self.device = device

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        data = self.user_data[idx]
        impression_id = data['ID']
        if len(data['history']) == 0 or len(data['candidate']) == 0:
            import pdb; pdb.set_trace()
        history = self.get_embedding(self.item_embeddings, data['history'])
        pos_item = self.get_embedding(self.item_embeddings, data['candidate'])
        return impression_id, history.to(self.device), pos_item.to(self.device), pos_item.to(self.device)



def infonce_loss(h_user, h_news, temperature=0.01):
    scores = F.sigmoid((h_user @ h_news.T)).squeeze()  # (5,)
    pos_score = torch.exp(scores[0] / temperature)
    neg_score = torch.sum(torch.exp(scores[1:] / temperature))
    loss = -torch.log(pos_score / (pos_score + neg_score))
    return loss