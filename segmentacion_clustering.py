"""
================================================================================
LABORATORIO 1 - SEGMENTACIÓN POR CLUSTERING EN IMÁGENES INDUSTRIALES/MECATRÓNICAS
================================================================================

REVISIÓN DEL ESTADO DEL ARTE
------------------------------
Este laboratorio implementa técnicas de segmentación por clustering con base en
las siguientes referencias recientes:

[1] Achanta et al. (2012) - SLIC Superpixels Compared to State-of-the-Art
    Superpixel Methods. IEEE TPAMI.
    Mejora: Introduce superpíxeles (SLIC) que agregan píxeles vecinos con
      características similares antes del clustering, incorporando información
      espacial de manera eficiente. Reduce el espacio de búsqueda y mejora
      la coherencia de bordes.

[2] Van den Bergh et al. (2012) / Chen et al. (2022) - SEEDS / Deep Spectral Methods
    Mejora: Los métodos espectrales y de grafos (Graph Cut / Normalized Cut)
      modelan la conectividad de píxeles como un grafo ponderado y optimizan
      la partición global. Permiten segmentación basada en afinidad en lugar
      de solo distancia euclidiana.

[3] Kirillov et al. (2023) - Segment Anything Model (SAM). ICCV 2023.
    Mejora: Modelo fundacional que utiliza embeddings profundos (ViT) pre-
      entrenados en >1B máscaras. Permite segmentación zero-shot y few-shot
      generalizando a dominios industriales sin reentrenamiento.

[4] Bezdek et al. (1984) / actualizado en múltiples trabajos modernos -
    Fuzzy C-Means Clustering.
    Mejora: Asignación suave (fuzzy) donde cada píxel tiene un grado de
      pertenencia a múltiples clusters, capturando incertidumbre en bordes
      y transiciones graduales típicas en imágenes industriales.

[5] Reynolds (2009) / Celeux & Govaert (1995) - Gaussian Mixture Models (GMM)
    Mejora: Modela cada cluster como una distribución gaussiana multivariada,
      permitiendo clusters de forma elíptica y densidad variable, más realista
      en espacios de color CIELab.

PIPELINE IMPLEMENTADO:
  1. Carga y preprocesamiento de imágenes industriales
  2. Conversión a RGB y CIELab
  3. Normalización y construcción de vector de características [color + espacial]
  4. K-Means con múltiples valores de k (evaluación con inercia y silueta)
  5. Fuzzy C-Means (FCM) comparativo
  6. Gaussian Mixture Model (GMM) comparativo
  7. Posprocesamiento: cierre morfológico y supresión de regiones pequeñas
  8. Visualización comparativa completa

Autores: Laboratorio IA - Noveno Semestre Mecatrónica
Fecha: 2026
================================================================================
"""

import os
import sys
import warnings
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

from skimage import io, color, morphology, measure
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import gaussian

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import scipy.ndimage as ndi

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
IMAGES_DIR = r"C:\Users\santi\OneDrive\MIO COLEGIO\Documentos\Noveno Semestre\IA\Lab1"
K_VALUES   = [2, 3, 4, 5, 6]          # Valores de k a probar
SPATIAL_WEIGHT = 0.15                  # Peso de coordenadas espaciales
FCM_MAX_ITER = 150                     # Iteraciones Fuzzy C-Means
RESIZE_MAX  = 256                      # Tamaño máximo de imagen para rapidez
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================
# UTILIDADES - CARGA Y PREPROCESAMIENTO
# ============================================================

def load_images(directory: str) -> list:
    """Carga todas las imágenes JPEG/PNG del directorio."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, ext)))
        paths.extend(glob.glob(os.path.join(directory, ext.upper())))
    paths = sorted(set(paths))

    images = []
    for p in paths:
        try:
            img = io.imread(p)
            # Convertir escala de grises o RGBA a RGB
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img = img[:, :, :3].astype(np.uint8)
            images.append({'path': p, 'name': os.path.basename(p), 'rgb': img})
            print(f"  OK  Cargada: {os.path.basename(p)}  [{img.shape[1]} x {img.shape[0]}]")
        except Exception as e:
            print(f"  ERROR cargando {p}: {e}")
    return images


def resize_image(img: np.ndarray, max_size: int = RESIZE_MAX) -> np.ndarray:
    """Redimensiona manteniendo aspecto si supera max_size."""
    from skimage.transform import resize as sk_resize
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return sk_resize(img, (new_h, new_w), anti_aliasing=True,
                     preserve_range=True).astype(np.uint8)


def build_feature_vector(img_rgb: np.ndarray,
                         use_lab: bool = True,
                         spatial_weight: float = SPATIAL_WEIGHT,
                         smooth_sigma: float = 1.0):
    """
    Construye el vector de características por pixel:
        [L*, a*, b*, row_norm, col_norm]   (CIELab + coordenadas espaciales)

    Retorna:
        features  : (H*W, 5) float32
        img_lab   : (H, W, 3) float32  -- imagen CIELab
        img_float : (H, W, 3) float32  -- imagen RGB float
    """
    img_float = img_as_float(img_rgb)
    # Suavizado suave para reducir ruido de sensor
    img_smooth = gaussian(img_float, sigma=smooth_sigma, channel_axis=-1)
    # Conversión CIELab
    img_lab = color.rgb2lab(img_smooth)
    H, W = img_rgb.shape[:2]

    # Coordenadas espaciales normalizadas [0, 1]
    rows = np.arange(H) / max(H - 1, 1)
    cols = np.arange(W) / max(W - 1, 1)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')   # (H, W)

    # Normalizar canales de color a [0, 1]
    L_norm = img_lab[:, :, 0] / 100.0
    a_norm = (img_lab[:, :, 1] + 128) / 255.0
    b_norm = (img_lab[:, :, 2] + 128) / 255.0
    color_feats = np.stack([L_norm, a_norm, b_norm], axis=-1)

    spatial_feats = np.stack([row_grid * spatial_weight,
                              col_grid * spatial_weight], axis=-1)

    combined = np.concatenate([color_feats, spatial_feats], axis=-1)   # (H, W, 5)
    features  = combined.reshape(-1, combined.shape[-1]).astype(np.float32)
    return features, img_lab, img_float


# ============================================================
# FUZZY C-MEANS (implementación propia)
# ============================================================

class FuzzyCMeans:
    """
    Fuzzy C-Means clustering (Bezdek, 1984).
    Parametros:
        n_clusters : número de clusters
        m          : exponente de fuzzificación (tipicamente 2)
        max_iter   : iteraciones máximas
        tol        : tolerancia de convergencia
    """
    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 max_iter: int = FCM_MAX_ITER, tol: float = 1e-4,
                 random_state: int = RANDOM_SEED):
        self.n_clusters   = n_clusters
        self.m            = m
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state
        self.centers_     = None
        self.U_           = None
        self.n_iter_      = 0

    def fit(self, X: np.ndarray):
        N, D = X.shape
        rng  = np.random.default_rng(self.random_state)
        U    = rng.random((N, self.n_clusters)).astype(np.float32)
        U   /= U.sum(axis=1, keepdims=True)

        for it in range(self.max_iter):
            U_m     = U ** self.m                                              # (N, C)
            centers = (U_m.T @ X) / (U_m.sum(axis=0)[:, None] + 1e-10)       # (C, D)
            diff    = X[:, None, :] - centers[None, :, :]                     # (N, C, D)
            dist2   = np.sum(diff**2, axis=-1) + 1e-10                        # (N, C)
            exp     = 2.0 / (self.m - 1)
            U_new   = np.zeros_like(U)
            for c in range(self.n_clusters):
                ratio     = dist2[:, c:c+1] / dist2
                U_new[:, c] = 1.0 / (ratio ** exp).sum(axis=1)
            delta = np.max(np.abs(U_new - U))
            U     = U_new
            self.n_iter_ = it + 1
            if delta < self.tol:
                break

        self.U_       = U
        self.centers_ = centers
        return self

    def predict(self) -> np.ndarray:
        """Retorna etiquetas hard (cluster de mayor pertenencia)."""
        return np.argmax(self.U_, axis=1)


# ============================================================
# POSPROCESAMIENTO
# ============================================================

def postprocess_mask(labels: np.ndarray,
                     min_size: int = 150,
                     closing_radius: int = 2) -> np.ndarray:
    """
    Posprocesamiento:
      1. Cierre morfológico para rellenar huecos pequeños
      2. Eliminar regiones menores a min_size píxeles
    """
    processed = labels.copy()
    n_clusters = labels.max() + 1
    selem = morphology.disk(closing_radius)

    for k in range(n_clusters):
        mask        = (processed == k)
        mask_closed = morphology.binary_closing(mask, selem)
        processed[mask_closed & ~mask] = k

    labeled_img = measure.label(processed, connectivity=2)
    for region in measure.regionprops(labeled_img):
        if region.area < min_size:
            rr, cc = region.coords[:, 0], region.coords[:, 1]
            r0 = max(0, rr.min()-3)
            r1 = min(processed.shape[0], rr.max()+4)
            c0 = max(0, cc.min()-3)
            c1 = min(processed.shape[1], cc.max()+4)
            patch = processed[r0:r1, c0:c1].flatten()
            vals, cnts = np.unique(patch, return_counts=True)
            own = processed[rr[0], cc[0]]
            mask_excl = vals != own
            if mask_excl.any():
                new_val = vals[mask_excl][cnts[mask_excl].argmax()]
            else:
                new_val = own
            processed[rr, cc] = new_val

    return processed


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_clustering(features: np.ndarray,
                         labels: np.ndarray,
                         sample_size: int = 5000) -> dict:
    """Calcula métricas de clustering (muestra para eficiencia)."""
    N = features.shape[0]
    if N > sample_size:
        idx     = np.random.choice(N, sample_size, replace=False)
        feats_s = features[idx]
        labels_s = labels[idx]
    else:
        feats_s, labels_s = features, labels

    metrics = {}
    n_unique = len(np.unique(labels_s))
    if n_unique > 1:
        try:
            metrics['silhouette']     = silhouette_score(feats_s, labels_s,
                                            sample_size=min(2000, N))
            metrics['davies_bouldin'] = davies_bouldin_score(feats_s, labels_s)
        except Exception:
            metrics['silhouette']     = np.nan
            metrics['davies_bouldin'] = np.nan
    else:
        metrics['silhouette']     = np.nan
        metrics['davies_bouldin'] = np.nan
    return metrics


# ============================================================
# VISUALIZACIÓN
# ============================================================

def colorize_labels(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Convierte mapa de etiquetas a imagen RGB de colores."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))[:n_clusters, :3]
    return (colors[labels] * 255).astype(np.uint8)


def plot_kmeans_k_comparison(img_rgb, results_k, img_name, save_dir):
    """Muestra segmentaciones K-Means para distintos k lado a lado."""
    n   = len(results_k)
    fig, axes = plt.subplots(2, n + 1, figsize=(4*(n+1), 8), facecolor='#1a1a2e')
    fig.suptitle(f'K-Means — Comparacion de k  |  {img_name}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)

    for row in range(2):
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title('Original', color='white', fontsize=10)
        axes[row, 0].axis('off')

    for i, (k, res) in enumerate(results_k.items()):
        col_img = colorize_labels(res['labels_2d'], k)
        axes[0, i+1].imshow(col_img)
        sil = res['metrics']['silhouette']
        axes[0, i+1].set_title(f'k={k}\nSil={sil:.3f}', color='white', fontsize=9)
        axes[0, i+1].axis('off')

        boundary = mark_boundaries(img_as_float(img_rgb), res['labels_2d'],
                                   color=(1, 0.8, 0), mode='thick')
        axes[1, i+1].imshow(boundary)
        axes[1, i+1].set_title(f'Bordes k={k}', color='white', fontsize=9)
        axes[1, i+1].axis('off')

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    plt.tight_layout(pad=0.5)
    safe = img_name.replace(' ', '_').replace('(', '').replace(')', '')
    out  = os.path.join(save_dir, f'kmeans_comparison_{safe}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f"    -> Guardado: {os.path.basename(out)}")


def plot_method_comparison(img_rgb, kmeans_res, fcm_res, gmm_res,
                            k, img_name, save_dir):
    """Compara K-Means vs FCM vs GMM para el mismo k."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='#0d0d1a')
    fig.suptitle(f'Comparacion de Metodos  |  k={k}  |  {img_name}',
                 color='white', fontsize=13, fontweight='bold')

    methods = [
        ('K-Means',       kmeans_res, '#4cc9f0'),
        ('Fuzzy C-Means', fcm_res,    '#f72585'),
        ('GMM',           gmm_res,    '#7bed9f'),
    ]

    for row in range(2):
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title('Original', color='white', fontsize=11, pad=6)
        axes[row, 0].axis('off')

    for col, (name, res, accent) in enumerate(methods, start=1):
        col_img  = colorize_labels(res['labels_2d'], k)
        boundary = mark_boundaries(img_as_float(img_rgb), res['labels_post_2d'],
                                   color=(1, 1, 0), mode='thick')
        sil = res['metrics']['silhouette']
        db  = res['metrics']['davies_bouldin']
        axes[0, col].imshow(col_img)
        axes[0, col].set_title(
            f'{name}\nSilueta={sil:.3f}  DB={db:.3f}',
            color=accent, fontsize=10, pad=6)
        axes[0, col].axis('off')

        axes[1, col].imshow(boundary)
        axes[1, col].set_title(f'{name} — post-proc', color=accent, fontsize=10, pad=6)
        axes[1, col].axis('off')

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_edgecolor('#222244')

    plt.tight_layout(pad=1.0)
    safe = img_name.replace(' ', '_').replace('(', '').replace(')', '')
    out  = os.path.join(save_dir, f'method_comparison_k{k}_{safe}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close()
    print(f"    -> Guardado: {os.path.basename(out)}")


def plot_metrics_summary(all_metrics: dict, save_dir: str):
    """Grafica resumen de métricas para todas las imágenes y k."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f0f23')
    fig.suptitle('Resumen de Metricas de Clustering — K-Means',
                 color='white', fontsize=13, fontweight='bold')

    n_imgs      = len(all_metrics)
    colors_list = plt.cm.plasma(np.linspace(0.2, 0.9, max(n_imgs, 1)))

    for ax in axes:
        ax.set_facecolor('#1a1a35')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334')

    for i, (img_name, metrics_by_k) in enumerate(all_metrics.items()):
        ks   = sorted(metrics_by_k.keys())
        sils = [metrics_by_k[k]['silhouette'] for k in ks]
        dbs  = [metrics_by_k[k]['davies_bouldin'] for k in ks]
        short = img_name[:30] + '...' if len(img_name) > 30 else img_name
        axes[0].plot(ks, sils, 'o-', color=colors_list[i], label=short, lw=2)
        axes[1].plot(ks, dbs,  's-', color=colors_list[i], label=short, lw=2)

    axes[0].set_title('Coeficiente de Silueta (mayor es mejor)',
                      color='white', fontsize=11)
    axes[0].set_xlabel('Numero de clusters k')
    axes[0].set_ylabel('Silueta')
    axes[0].legend(fontsize=8, facecolor='#1a1a35', labelcolor='white',
                   loc='lower right')
    axes[0].grid(alpha=0.2, color='#445')

    axes[1].set_title('Davies-Bouldin (menor es mejor)',
                      color='white', fontsize=11)
    axes[1].set_xlabel('Numero de clusters k')
    axes[1].set_ylabel('Davies-Bouldin')
    axes[1].legend(fontsize=8, facecolor='#1a1a35', labelcolor='white',
                   loc='upper right')
    axes[1].grid(alpha=0.2, color='#445')

    plt.tight_layout()
    out = os.path.join(save_dir, 'metrics_summary.png')
    plt.savefig(out, dpi=120, bbox_inches='tight',
                facecolor='#0f0f23', edgecolor='none')
    plt.close()
    print(f"\n  -> Resumen de metricas guardado: {os.path.basename(out)}")


def plot_feature_space(features: np.ndarray, labels: np.ndarray,
                       img_name: str, k: int, save_dir: str):
    """Visualiza espacio de características en 2D con PCA."""
    N      = features.shape[0]
    idx    = np.random.choice(N, min(4000, N), replace=False)
    pca    = PCA(n_components=2, random_state=RANDOM_SEED)
    f2d    = pca.fit_transform(features[idx])
    cmap   = ListedColormap(plt.cm.tab10(np.linspace(0, 1, max(k, 10)))[:k])

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    sc = ax.scatter(f2d[:, 0], f2d[:, 1], c=labels[idx],
                    cmap=cmap, s=5, alpha=0.6, linewidths=0)
    ax.set_title(f'Espacio de características PCA 2D\n{img_name}  k={k}',
                 color='white', fontsize=11)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334')
    plt.colorbar(sc, ax=ax, label='Cluster')
    plt.tight_layout()
    safe = img_name.replace(' ', '_').replace('(', '').replace(')', '')
    out  = os.path.join(save_dir, f'feature_space_{safe}_k{k}.png')
    plt.savefig(out, dpi=110, bbox_inches='tight',
                facecolor='#0d0d1a', edgecolor='none')
    plt.close()


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run_pipeline():
    print("=" * 70)
    print("  SEGMENTACION POR CLUSTERING — LABORATORIO IA")
    print("=" * 70)
    print()

    # ── 1. Cargar imagenes ─────────────────────────────────────────────────
    print("PASO 1: Carga de imagenes")
    images = load_images(IMAGES_DIR)
    if not images:
        print("  ERROR: No se encontraron imagenes. Verifica la ruta.")
        sys.exit(1)
    print(f"  Total imagenes cargadas: {len(images)}\n")

    results_dir = os.path.join(IMAGES_DIR, 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    print(f"  Resultados se guardaran en: {results_dir}\n")

    all_metrics = {}  # {img_name: {k: metrics}}

    for img_data in images:
        name     = img_data['name']
        img_orig = img_data['rgb']

        print("-" * 70)
        print(f"  IMAGEN: {name}")
        print("-" * 70)

        img_rgb = resize_image(img_orig, RESIZE_MAX)
        H, W    = img_rgb.shape[:2]
        N       = H * W
        print(f"  Dimensiones procesadas: {W} x {H}  ({N} pixeles)")

        # ── 2. Preparacion de datos ────────────────────────────────────────
        print("\nPASO 2: Preparacion de datos")
        features, img_lab, img_float = build_feature_vector(
            img_rgb, use_lab=True, spatial_weight=SPATIAL_WEIGHT)
        print(f"  Vector de caracteristicas: {features.shape}")
        print(f"  Canales: [L*, a*, b*, row_norm, col_norm]  (CIELab + espacial)")

        # ── 3. K-Means con distintos k ─────────────────────────────────────
        print("\nPASO 3: K-Means (multiples k)")
        results_k    = {}
        metrics_by_k = {}

        for k in K_VALUES:
            t0 = time.time()
            km = KMeans(n_clusters=k, init='k-means++', n_init=5,
                        max_iter=300, random_state=RANDOM_SEED)
            km.fit(features)
            labels_flat  = km.labels_
            labels_2d    = labels_flat.reshape(H, W)
            labels_post  = postprocess_mask(labels_2d, min_size=max(50, N//200))
            mets         = evaluate_clustering(features, labels_flat)
            elapsed      = time.time() - t0

            results_k[k] = {
                'labels_2d':      labels_2d,
                'labels_post_2d': labels_post,
                'centers':        km.cluster_centers_,
                'inertia':        km.inertia_,
                'metrics':        mets,
            }
            metrics_by_k[k] = mets
            print(f"  k={k}: inercia={km.inertia_:.1f}  "
                  f"silueta={mets['silhouette']:.3f}  "
                  f"DB={mets['davies_bouldin']:.3f}  "
                  f"({elapsed:.1f}s)")

        all_metrics[name] = metrics_by_k

        # Mejor k segun silueta
        best_k = max(
            metrics_by_k,
            key=lambda x: metrics_by_k[x]['silhouette']
                          if not np.isnan(metrics_by_k[x]['silhouette']) else -1
        )
        print(f"\n  Mejor k (Silueta): {best_k}")

        # Visualizaciones K-Means
        plot_kmeans_k_comparison(img_rgb, results_k, name, results_dir)
        plot_feature_space(features,
                           results_k[best_k]['labels_2d'].flatten(),
                           name, best_k, results_dir)

        # ── 4. Fuzzy C-Means ──────────────────────────────────────────────
        print(f"\nPASO 4: Fuzzy C-Means  (k={best_k})")
        t0  = time.time()
        fcm = FuzzyCMeans(n_clusters=best_k, m=2.0,
                          max_iter=FCM_MAX_ITER, random_state=RANDOM_SEED)
        fcm.fit(features)
        fcm_labels_flat = fcm.predict()
        fcm_labels_2d   = fcm_labels_flat.reshape(H, W)
        fcm_post        = postprocess_mask(fcm_labels_2d,
                                           min_size=max(50, N//200))
        fcm_mets        = evaluate_clustering(features, fcm_labels_flat)
        print(f"  FCM converge en {fcm.n_iter_} iters  ({time.time()-t0:.1f}s)")
        print(f"  silueta={fcm_mets['silhouette']:.3f}  "
              f"DB={fcm_mets['davies_bouldin']:.3f}")
        fcm_res = {
            'labels_2d':      fcm_labels_2d,
            'labels_post_2d': fcm_post,
            'metrics':        fcm_mets,
        }

        # ── 5. GMM ────────────────────────────────────────────────────────
        print(f"\nPASO 5: Gaussian Mixture Model  (k={best_k})")
        t0  = time.time()
        gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                              n_init=3, max_iter=200, random_state=RANDOM_SEED)
        gmm.fit(features)
        gmm_labels_flat = gmm.predict(features)
        gmm_labels_2d   = gmm_labels_flat.reshape(H, W)
        gmm_post        = postprocess_mask(gmm_labels_2d,
                                           min_size=max(50, N//200))
        gmm_mets        = evaluate_clustering(features, gmm_labels_flat)
        print(f"  GMM converge en {gmm.n_iter_} iters  ({time.time()-t0:.1f}s)")
        print(f"  silueta={gmm_mets['silhouette']:.3f}  "
              f"DB={gmm_mets['davies_bouldin']:.3f}")
        print(f"  AIC={gmm.aic(features):.1f}  BIC={gmm.bic(features):.1f}")
        gmm_res = {
            'labels_2d':      gmm_labels_2d,
            'labels_post_2d': gmm_post,
            'metrics':        gmm_mets,
        }

        # ── 6. Comparacion de metodos ──────────────────────────────────────
        print(f"\nPASO 6: Generando visualizaciones comparativas...")
        plot_method_comparison(img_rgb,
                               results_k[best_k], fcm_res, gmm_res,
                               best_k, name, results_dir)
        print()

    # ── Resumen global ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESUMEN GLOBAL DE METRICAS")
    print("=" * 70)
    plot_metrics_summary(all_metrics, results_dir)

    print(f"\n  {'Imagen':<35} {'k':<4} {'Silueta':>10} {'DB':>10}")
    print(f"  {'-'*62}")
    for img_name, mets_k in all_metrics.items():
        for k, mets in sorted(mets_k.items()):
            sil   = f"{mets['silhouette']:.4f}" if not np.isnan(mets['silhouette']) else '   NaN'
            db    = f"{mets['davies_bouldin']:.4f}" if not np.isnan(mets['davies_bouldin']) else '   NaN'
            short = img_name[:33]
            print(f"  {short:<35} {k:<4} {sil:>10} {db:>10}")

    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETADO")
    print(f"  Resultados en: {results_dir}")
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline()
