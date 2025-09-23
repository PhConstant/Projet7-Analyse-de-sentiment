## postprocess_data.py
# Source/postprocess_data.py
# Fonctions de post-traitement des sorties de modèles
# Auteur : Philippe CONSTANT
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt


## Fonction de post-traitement des expériences MLflow
def postprocess_model_output(y, y_pred, y_pred_proba):
    """
    Post-traitement des sorties du modèle.
    """
    report = classification_report(y, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    roc_auc = roc_auc_score(y, y_pred_proba)

    output_dict = {
            "Accuracy": report['accuracy'],
            "F1_negatif": report['Negative']['f1-score'],
            "F1_positif": report['Positive']['f1-score'],
            "Recall_negatif": report['Negative']['recall'],
            "Recall_positif": report['Positive']['recall'],
            "Precision_negatif": report['Negative']['precision'],
            "Precision_positif": report['Positive']['precision'],
            "ROC_AUC": roc_auc
        }
    return output_dict


def plot_training_history(history, show=True):
    """
    Affiche l'évolution des métriques d'entraînement et de validation
    dans une seule figure avec des subplots.

    Paramètres
    ----------
    history : keras.callbacks.History
        Objet retourné par model.fit()
    """
    history_dict = history.history
    epochs = range(1, len(next(iter(history_dict.values()))) + 1)

    # On récupère uniquement les métriques côté entraînement
    metrics = [m for m in history_dict.keys() if not m.startswith('val_')]

    # Création des subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    # Si une seule métrique, axes n'est pas un tableau
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(epochs, history_dict[metric], 'o-', label=f'Train {metric}')
        
        val_key = f'val_{metric}'
        if val_key in history_dict:
            ax.plot(epochs, history_dict[val_key], 'o-', label=f'Validation {metric}')
        
        ax.set_title(f'Évolution de {metric}')
        ax.set_xlabel('Époques')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if show:
        plt.show()
    return fig
