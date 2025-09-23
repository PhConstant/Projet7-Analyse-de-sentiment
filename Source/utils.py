## utils.py
# Source/utils.py
# Fonctions utilitaires diverses
# Auteur : Philippe CONSTANT
def convert_param_to_string(param):
    # Si c'est déjà une string
    if isinstance(param, str):
        return param
    
    # Si c'est une liste de stopwords on ajoute juste la longueur de la liste
    if isinstance(param, list):
        return f"stopwords[{len(param)}]"
    
    # Si c'est une fonction ou une méthode
    if callable(param):
        # Pour une bound method on va juste retourner le nom de la classe parente
        if hasattr(param, "__self__") and param.__self__ is not None:
            return param.__self__.__class__.__name__
        # Pour une fonction classique ou une méthode statique
        return param.__name__

    # Pour un objet de classe (ex: CountVectorizer, PorterStemmer)
    if hasattr(param, "__class__"):
        return param.__class__.__name__

    # fallback
    return str(param)
