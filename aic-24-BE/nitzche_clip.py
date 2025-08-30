import os
from pathlib import Path
import numpy as np
import json
import pickle
import csv
import torch
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer  # type: ignore
from tqdm import tqdm


class NitzcheCLIP:
    def __init__(self, feature_path):
        self.file_path_list, self.image_feature, self.youtube_link, self.fps = [], [], {}, {}
        be_root = Path(__file__).resolve().parent
        
        # Validate feature directory exists
        if not os.path.exists(feature_path):
            raise ValueError(f"Feature directory does not exist: {feature_path}")
        
        feature_files = sorted(os.listdir(feature_path))
        if not feature_files:
            raise ValueError(f"No feature files found in: {feature_path}")
        
        for filename in tqdm(feature_files):
            filepath = os.path.join(feature_path, filename)
            with open(filepath, "rb") as fp:
                file_path, image_feature = pickle.load(fp)
                
                self.file_path_list.extend(file_path)
                self.image_feature.append(image_feature)
        
        # Try multiple possible locations for media-info and map-keyframes
        possible_base_dirs = [
            be_root / 'data',  # Original location: aic-24-BE/data/
            be_root.parent / 'example_dataset',  # Dataset location: example_dataset/
            Path.cwd() / 'example_dataset',  # Current working directory
            Path.cwd().parent / 'example_dataset',  # Parent directory
        ]
        
        meta_dir = None
        map_dir = None
        
        # Find media-info directory
        for base_dir in possible_base_dirs:
            potential_meta = base_dir / 'media-info'
            if potential_meta.is_dir():
                meta_dir = potential_meta
                break
        
        # Find map-keyframes directory
        for base_dir in possible_base_dirs:
            potential_map = base_dir / 'map-keyframes'
            if potential_map.is_dir():
                map_dir = potential_map
                break
        
        # Validate media-info directory exists
        if not meta_dir or not meta_dir.is_dir():
            searched_paths = [str(base_dir / 'media-info') for base_dir in possible_base_dirs]
            raise ValueError(f"Media info directory not found in any of: {searched_paths}")
        
        # First, try to load FPS from keyframe mapping files (most reliable source)
        fps_from_keyframes = {}
        if map_dir and map_dir.is_dir():
            for csv_file in map_dir.glob('*.csv'):
                video_id = csv_file.stem
                try:
                    with csv_file.open('r', encoding='utf-8') as f:
                        rdr = csv.reader(f)
                        rows = list(rdr)
                        if len(rows) > 1:  # Has header and at least one data row
                            header = rows[0]
                            # Find FPS column index
                            fps_idx = None
                            for i, col in enumerate(header):
                                if col.strip().lower() == 'fps':
                                    fps_idx = i
                                    break
                            if fps_idx is not None:
                                # Get FPS from first data row (all rows should have same FPS)
                                fps_val = float(rows[1][fps_idx])
                                if fps_val > 0:
                                    fps_from_keyframes[video_id] = fps_val
                except Exception as e:
                    print(f"Warning: Could not read FPS from {csv_file}: {e}")
        
        # Now process media-info files
        for filename in tqdm(sorted(os.listdir(meta_dir))):
            filepath = meta_dir / filename
            video_id = filename.split('.')[0]
            with open(filepath, 'r') as file:
                data = json.load(file)
            # watch_url is required by the UI; fall back to empty string if absent
            url = data.get('watch_url') or data.get('url') or ''
            self.youtube_link[video_id] = url
            
            # Priority: 1) keyframe mapping FPS, 2) media-info FPS, 3) default 25
            if video_id in fps_from_keyframes:
                fps_val = fps_from_keyframes[video_id]
            else:
                # Try to get from media-info
                try:
                    fps_val = float(data.get('fps', 25))
                    if fps_val <= 0:
                        fps_val = 25.0
                except Exception:
                    fps_val = 25.0
            
            self.fps[video_id] = fps_val
        
        self.image_feature = np.concatenate(self.image_feature, axis=0)
        self.model, self.processor = self._load_model()

    def _load_model(self):
        device = os.environ.get("CLIP_DEVICE", "cpu").lower()
        if device not in {"cpu", "cuda"}:
            device = "cpu"
        # Configurable model + pretrained to match recomputed image features
        model_name = os.environ.get("CLIP_MODEL_NAME", "ViT-B-32")
        pretrained = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
        # create text+image towers; we use only text encode at runtime
        model, _ = create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = get_tokenizer(model_name)
        return model, tokenizer

    def featurize_text(self, text_query):
        device = next(self.model.parameters()).device
        # Tokenizer accepts list of strings
        if isinstance(text_query, str):
            texts = [text_query]
        else:
            texts = text_query
        text_input = self.processor(texts, context_length=self.model.context_length).to(device)
        
        features_text = self.model.encode_text(text_input)
        features_text = F.normalize(features_text, dim=-1)
        return features_text.cpu().detach().numpy()

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
            
        # Find the video frames directory (flexible path resolution)
        be_root = Path(__file__).resolve().parent
        frames_dirs = [
            be_root / 'data' / 'video_frames',  # Original location
            be_root.parent / 'aic-24-BE' / 'data' / 'video_frames',  # Alternative backend location
            Path.cwd() / 'aic-24-BE' / 'data' / 'video_frames',  # Current working directory
        ]
        
        frames_dir = None
        for potential_frames in frames_dirs:
            if potential_frames.exists():
                frames_dir = potential_frames
                break
        
        if not frames_dir:
            frames_dir = be_root / 'data' / 'video_frames'  # Fallback to original
        
        return [
            {
                'img_path': str(frames_dir / vid / timeframe[0]),
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
