from insightface.app import FaceAnalysis
from pathlib import Path
from tqdm import tqdm
import pickle
import cv2

app = None


def initialize_models():
    global app
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))


def load_gallery_embeddings(gallery_path):
    gallery_path = Path(gallery_path)
    img_paths = list(gallery_path.glob('*.jpg'))

    failed_images = []
    no_face_images = []
    embeddings = []
    image_paths = []

    for img_path in tqdm(img_paths, desc="Processando galeria"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                failed_images.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'reason': 'Falha ao carregar com cv2.imread'
                })
                continue

            faces = app.get(img)
            if len(faces) > 0:
                embedding = faces[0].embedding
                embeddings.append(embedding)
                image_paths.append(str(img_path))
            else:
                no_face_images.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'image_size': f"{img.shape[1]}x{img.shape[0]}"
                })
        except Exception as e:
            failed_images.append({
                'path': str(img_path),
                'filename': img_path.name,
                'reason': str(e)
            })

    data = {
        'embeddings': embeddings,
        'paths': image_paths,
    }

    print(f'Imagens processadas: {len(embeddings)}')
    print(f'Imagens sem faces: {len(no_face_images)}')
    print(f'Imagens falhadas: {len(failed_images)}')

    with open('gallery_embeddings.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data


def main():
    initialize_models()
    load_gallery_embeddings("../../nobackup3/face_rec_pep/dataset_pep/image_gallery")


if __name__ == "__main__":
    main()
