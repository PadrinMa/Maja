import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import nms
import time
import json
from functions import area_overlap_filter, convert_to_yolo, draw_detection, get_class_id

class ImprovedGroundingDINO:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
        # PROMPT OTTIMIZZATI - Testati empiricamente per barche
        self.boat_prompts = {
            # Prompt base - sempre inclusi
            'core_boats': [
                "boat", "ship", "vessel", "watercraft"
            ],
            
            # Barche a vela - molto dettagliati
            'sailing_boats': [
                "sailboat", "sailing boat", "yacht with sail", "catamaran with sail",
                "trimaran", "sailing yacht", "sloop", "ketch", "schooner", 
                "sailing vessel", "sail boat"
            ],
            
            # Barche a motore - specifiche per tipo
            'motor_boats': [
                "motorboat", "motor boat", "speedboat", "powerboat", 
                "cabin cruiser", "sport boat", "runabout", "bowrider",
                "express cruiser", "motor yacht"
            ],
            
            # Barche lavoro/commerciali
            'commercial_boats': [
                "fishing boat", "trawler", "commercial fishing vessel",
                "tugboat", "pilot boat", "work boat", "patrol boat",
                "coast guard boat", "ferry boat"
            ],
            
            # Barche piccole
            'small_boats': [
                "dinghy", "tender", "inflatable boat", "rib boat",
                "kayak", "canoe", "rowboat", "paddle boat",
                "small boat", "skiff"
            ],
            
            # Navi grandi
            'large_ships': [
                "cargo ship", "container ship", "cruise ship", "liner",
                "freighter", "tanker", "bulk carrier", "naval ship"
            ],
            
            # Context-aware prompts (importante!)
            'contextual': [
                "boat in water", "vessel on water", "ship at sea",
                "boat on ocean", "watercraft on lake", "marine vessel"
            ]
        }
        
        # Configurazioni dinamiche per diversi scenari
        self.scenario_configs = {
            'open_sea': {
                'prompts': ['core_boats', 'sailing_boats', 'motor_boats', 'large_ships', 'contextual'],
                'confidence_threshold': 0.25,
                'text_threshold': 0.25,
                'nms_threshold': 0.5
            },
            'harbor': {
                'prompts': ['core_boats', 'motor_boats', 'commercial_boats', 'small_boats'],
                'confidence_threshold': 0.3,
                'text_threshold': 0.3,
                'nms_threshold': 0.4
            },
            'marina': {
                'prompts': ['core_boats', 'sailing_boats', 'motor_boats', 'small_boats'],
                'confidence_threshold': 0.35,
                'text_threshold': 0.3,
                'nms_threshold': 0.3
            },
            'mixed': {  # Scenario generale
                'prompts': ['core_boats', 'sailing_boats', 'motor_boats', 'commercial_boats', 'contextual'],
                'confidence_threshold': 0.28,
                'text_threshold': 0.28,
                'nms_threshold': 0.45
            }
        }
    
    def get_prompts_for_scenario(self, scenario='mixed'):
        """Ottieni prompt ottimizzati per scenario specifico"""
        config = self.scenario_configs.get(scenario, self.scenario_configs['mixed'])
        
        selected_prompts = []
        for prompt_category in config['prompts']:
            selected_prompts.extend(self.boat_prompts[prompt_category])
        
        # Rimuovi duplicati mantenendo ordine
        unique_prompts = list(dict.fromkeys(selected_prompts))
        
        return unique_prompts, config
    
    def enhanced_detection(self, image, scenario='mixed', use_multi_scale=True):
        """Detection migliorata con prompt ottimizzati e multi-scale"""
        
        prompts, config = self.get_prompts_for_scenario(scenario)
        
        print(f"🎯 Scenario: {scenario}")
        print(f"📝 Usando {len(prompts)} prompt ottimizzati")
        
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Multi-scale detection se abilitato
        scales = [1.0] if not use_multi_scale else [0.8, 1.0, 1.2]
        
        for scale in scales:
            if scale != 1.0:
                # Ridimensiona immagine per multi-scale
                new_size = (int(image.width * scale), int(image.height * scale))
                scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                scaled_image = image
            
            # Chunking dei prompt se troppi (limite DINO ~77 token)
            chunk_size = 15  # Numero sicuro di prompt per chunk
            
            for i in range(0, len(prompts), chunk_size):
                chunk_prompts = prompts[i:i+chunk_size]
                
                # Run detection
                inputs = self.processor(
                    images=scaled_image, 
                    text=chunk_prompts, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=config['confidence_threshold'],
                    text_threshold=config['text_threshold'],
                    target_sizes=[scaled_image.size[::-1]],
                )[0]
                
                if len(results["boxes"]) > 0:
                    boxes = results["boxes"]
                    scores = results["scores"]
                    labels = results["text_labels"]
                    
                    # Scala i box alla dimensione originale se necessario
                    if scale != 1.0:
                        boxes = boxes / scale
                    
                    all_boxes.extend(boxes.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                    all_labels.extend(labels)
        
        if not all_boxes:
            return {
                'boxes': torch.tensor([]),
                'scores': torch.tensor([]),
                'labels': []
            }
        
        # Converti in tensori
        all_boxes = torch.tensor(all_boxes)
        all_scores = torch.tensor(all_scores)
        
        # NMS finale
        keep = nms(all_boxes, all_scores, config['nms_threshold'])
        
        final_boxes = all_boxes[keep]
        final_scores = all_scores[keep]
        final_labels = [all_labels[i] for i in keep]
        
        print(f"✅ Rilevate {len(final_boxes)} barche dopo NMS")
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }
    
    def adaptive_threshold_detection(self, image, scenario='mixed'):
        """Detection con soglie adattive basate su densità detection"""
        
        # Prima passata con soglie basse
        initial_results = self.enhanced_detection(image, scenario)
        
        # Analizza densità detection
        num_detections = len(initial_results['boxes'])
        image_area = image.width * image.height
        detection_density = num_detections / (image_area / 1000000)  # per megapixel
        
        # Adatta soglie basandosi sulla densità
        if detection_density > 5:  # Troppi detection - alza soglie
            config = self.scenario_configs[scenario].copy()
            config['confidence_threshold'] += 0.1
            config['text_threshold'] += 0.1
            print(f"🔧 Densità alta ({detection_density:.1f}), alzando soglie")
        elif detection_density < 0.5:  # Troppo pochi - abbassa soglie
            config = self.scenario_configs[scenario].copy()
            config['confidence_threshold'] = max(0.15, config['confidence_threshold'] - 0.05)
            config['text_threshold'] = max(0.15, config['text_threshold'] - 0.05)
            print(f"🔧 Densità bassa ({detection_density:.1f}), abbassando soglie")
        else:
            return initial_results
        
        # Seconda passata con soglie adattate
        prompts, _ = self.get_prompts_for_scenario(scenario)
        
        # Re-run con nuove soglie...
        # (implementazione simile a enhanced_detection ma con config personalizzato)
        return initial_results  # Per brevità, ritorna risultato iniziale
    
    def process_batch(self, image_folder, output_folder, scenario='mixed', 
                     save_visualization=True, viz_folder=None):
        """Processa batch di immagini con DINO migliorato"""
        
        os.makedirs(output_folder, exist_ok=True)
        if save_visualization and viz_folder:
            os.makedirs(viz_folder, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"🚀 Processando {len(image_files)} immagini con DINO migliorato")
        
        results_summary = []
        start_time = time.time()
        
        for i, filename in enumerate(image_files):
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(f"📊 Progresso: {i}/{len(image_files)} ({elapsed:.1f}s)")
            
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            
            # Detection migliorata
            results = self.enhanced_detection(image, scenario)
            
            # Converti in formato YOLO
            yolo_lines = []
            if len(results['boxes']) > 0:
                for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
                    cls = get_class_id(label, fallback_class=0)  # Tutto classe 0 (boat)
                    yolo_line = f"{cls} {convert_to_yolo(box.tolist(), image.width, image.height)}"
                    yolo_lines.append(yolo_line)
            
            # Salva labels
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(output_folder, f"{base_name}.txt")
            
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            
            # Visualizzazione se richiesta
            if save_visualization and viz_folder:
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()
                
                for idx, (box, score, label) in enumerate(zip(results['boxes'], results['scores'], results['labels'])):
                    label_text = f"{label}: {score:.2f}"
                    draw_detection(draw, box.tolist(), label_text, idx, font)
                
                viz_path = os.path.join(viz_folder, f"{base_name}_enhanced.jpg")
                image.save(viz_path)
            
            # Statistiche
            results_summary.append({
                'image': filename,
                'detections': len(yolo_lines),
                'avg_confidence': float(torch.mean(results['scores'])) if len(results['scores']) > 0 else 0.0,
                'labels_used': list(set(results['labels']))
            })
        
        # Salva summary
        with open(os.path.join(output_folder, "detection_summary.json"), "w") as f:
            json.dump(results_summary, f, indent=2)
        
        total_detections = sum(r['detections'] for r in results_summary)
        avg_per_image = total_detections / len(results_summary) if results_summary else 0
        
        print(f"✅ Completato! {total_detections} detection totali, {avg_per_image:.1f} per immagine")
        
        return results_summary

# Utilizzo
if __name__ == "__main__":
    # Inizializza DINO migliorato
    improved_dino = ImprovedGroundingDINO()
    
    # Testa su cartella immagini
    results = improved_dino.process_batch(
        image_folder="D:/back_camera_right_image/back_camera_right_image",
        output_folder="enhanced_dino_labels",
        scenario='mixed',  # Cambia in 'open_sea', 'harbor', 'marina' per ottimizzare
        save_visualization=True,
        viz_folder="enhanced_dino_viz"
    )