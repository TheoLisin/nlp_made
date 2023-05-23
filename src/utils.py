import torch
from tqdm.notebook import tqdm
from nltk.translate.bleu_score import corpus_bleu

from pl_utils.pl_model import ModelWrapper
from pl_utils.pl_dataset import PlTranslationDataset


def calc_blue(model: ModelWrapper, pl_dataset: PlTranslationDataset, device: torch.device):
    dataloader = pl_dataset.test_dataloader()
    model.eval()
    generated_corpa = []
    target_corpa = []
    batch_first = False

    with torch.no_grad():
        for (source, target) in tqdm(dataloader):
            translation = model.forward((source.to(device), target.to(device)), 0, teacher_forcing_ratio=0)
            translation = translation.argmax(dim=-1).cpu().numpy()

            if not batch_first:
                translation = translation.T
                target = target.T

            for gen, orig in zip(translation, target):
                dec_gen = pl_dataset.target_lang.decode(gen)
                dec_orig = pl_dataset.target_lang.decode(orig)
                generated_corpa.append(dec_gen)
                target_corpa.append([dec_orig])
    
    print("Generated sample: ", " ".join(generated_corpa[0]))
    print("Target sample: ", " ".join(target_corpa[0][0]))
    
    return corpus_bleu(target_corpa, generated_corpa) * 100