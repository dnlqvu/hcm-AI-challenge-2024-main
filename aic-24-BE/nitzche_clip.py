import os
from pathlib import Path
import numpy as np
import json
import pickle
import csv
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer 
from tqdm import tqdm


class NitzcheCLIP:
    def __init__(self, feature_path):
        self.file_path_list, self.image_feature, self.youtube_link, self.fps = [], [], {}, {}
        be_root = Path(__file__).resolve().parent
        for filename in tqdm(sorted(os.listdir(feature_path))):
            filepath = os.path.join(feature_path, filename)
            with open(filepath, "rb") as fp:
                file_path, image_feature = pickle.load(fp)
                
                self.file_path_list.extend(file_path)
                self.image_feature.append(image_feature)
        
        meta_dir = be_root / 'data' / 'media-info'
        map_dir = be_root / 'data' / 'map-keyframes'
        for filename in tqdm(sorted(os.listdir(meta_dir))):
            filepath = meta_dir / filename
            video_id = filename.split('.')[0]
            with open(filepath, 'r') as file:
                data = json.load(file)
            # watch_url is required by the UI; fall back to empty string if absent
            url = data.get('watch_url') or data.get('url') or ''
            self.youtube_link[video_id] = url
            # fps is sometimes missing in some media-info exports; default to 25
            try:
                fps_val = float(data.get('fps', 25))
                if fps_val <= 0:
                    fps_val = 25.0
            except Exception:
                fps_val = 25.0
            # Fallback to map-keyframes CSV if available and media-info lacks fps
            if (not data.get('fps')) and map_dir.is_dir():
                csv_path = map_dir / f"{video_id}.csv"
                if csv_path.exists():
                    try:
                        with csv_path.open('r', encoding='utf-8') as f:
                            rdr = csv.reader(f)
                            rows = list(rdr)
                            if rows:
                                header = rows[0]
                                has_header = any(not c.replace('.', '', 1).isdigit() for c in header)
                                start = 1 if has_header else 0
                                fps_idx = None
                                if has_header:
                                    lower = [h.strip().lower() for h in header]
                                    for i, h in enumerate(lower):
                                        if h == 'fps':
                                            fps_idx = i
                                            break
                                # take first nonzero fps encountered
                                for row in rows[start:]:
                                    try:
                                        val = float(row[fps_idx or 2])  # common index 2 when columns are n,pts_time,fps,frame_idx
                                    except Exception:
                                        continue
                                    if val > 0:
                                        fps_val = val
                                        break
                    except Exception:
                        pass
            self.fps[video_id] = fps_val
        
        self.image_feature = np.concatenate(self.image_feature, axis=0)
        self.model, self.processor = self._load_model()

    def _load_model(self):
        device = "cpu"
        # Align runtime text encoder with ViT-B/32 image features
        # Pretrained weights: LAION2B s34B b79K
        model, _ = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K', device=device)
        txt_processors = get_tokenizer('ViT-B-32')
        return model, txt_processors

    def featurize_text(self, text_query):
        device = "cpu"
        text_input = self.processor(text_query, context_length=self.model.context_length).to(device)
        
        features_text = self.model.encode_text(text_input)
        features_text = F.normalize(features_text, dim=-1)
        features_text = features_text.cpu().detach().numpy()

        return features_text

    def predict(self, text_query, top=500):
        features_text = self.featurize_text(text_query)    
        results = np.squeeze(np.matmul(self.image_feature, features_text.T))
        
        results = np.argsort(results)[::-1].tolist()
        sorted_path_list = [self.file_path_list[index] for index in results[:top]]
        
        results_dict = {}
        for path_list in sorted_path_list:
            vid, timeframe = path_list.split('/')[-2:]
            
            if vid not in results_dict:
                results_dict.setdefault(vid, [])
                results_dict[vid].append((timeframe, True))
            else:
                results_dict[vid].append((timeframe, False))
            results_dict[vid] = sorted(results_dict[vid], key=lambda x: int(x[0][:-4]))
            
        return [
            {
                'img_path': os.path.join('./data/video_frames', vid, timeframe[0]),
                'youtube_link': self._with_time_param(self.youtube_link[vid], int(int(timeframe[0].split('.')[0]) / self.fps[vid])),
                'fps': self.fps[vid],
                'highlight': timeframe[1]
            } 
            for vid, timeframe_list in results_dict.items() for timeframe in timeframe_list
        ]

    @staticmethod
    def _with_time_param(url: str, seconds: int) -> str:
        if not url:
            return ''
        sep = '&' if '?' in url else '?'
        return f"{url}{sep}t={seconds}s"
        
    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))
    
if __name__ == '__main__':
    # nitzche_ins = NitzcheCLIP(feature_path='./data/clip_vit_h14_features')
    # nitzche_ins.save('./data/clip_vit_h14_nitzche.pkl')
    
    # Test function matching    
    nitzche_ins = NitzcheCLIP.load('./data/clip_vit_h14_nitzche.pkl')
    image_paths = nitzche_ins.predict(text_query="A shot from a camera on a car filming the journey. In the shot there is a yellow sign \"COM BINH DAN\" with red letters. Next the camera switches to another section of the road. In the next shot there is a person wearing a black shirt with a pink suitcase standing on the right hand side.")
    breakpoint()
