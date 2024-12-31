import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def estimate_propensity_scores(df, treatment_col, covariate_cols):
    model = LogisticRegression()
    model.fit(df[covariate_cols], df[treatment_col])
    propensity_scores = model.predict_proba(df[covariate_cols])[:, 1]
    return propensity_scores

def perform_matching(df, treatment_col, covariate_cols, factor_vars, 
                     treatment_value=1, caliper=0.05):
    """
    Realiza 1:1 matching sin reemplazo con caliper bidireccional,
    eliminando tanto tratados como controles que no encuentren match.
    """

    # Codificamos variables categóricas para LogisticRegression
    for col in factor_vars:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 1. Calculamos los propensity scores
    propensity_scores = estimate_propensity_scores(df, treatment_col, covariate_cols)
    df["propensity_score"] = propensity_scores

    # 2. Separamos en tratados y controles
    treated = df[df[treatment_col] == treatment_value].copy()
    control = df[df[treatment_col] != treatment_value].copy()

    # 3. Ordenamos cada grupo por su propensity_score
    treated = treated.sort_values(by="propensity_score").reset_index(drop=True)
    control = control.sort_values(by="propensity_score").reset_index(drop=True)

    # 4. Identificar quién es el grupo minoritario y quién el mayoritario
    #    para recorrer sólo el menor y buscar match en el mayor.
    if len(treated) <= len(control):
        smaller_group = treated
        larger_group = control
        label_smaller = "treated"
    else:
        smaller_group = control
        larger_group = treated
        label_smaller = "control"

    matched_smaller = []
    matched_larger = []

    # 5. Realizamos el matching
    # Usamos dos índices i (para el menor) y j (para el mayor).
    j = 0
    for i in range(len(smaller_group)):
        score_i = smaller_group.loc[i, "propensity_score"]
        # Avanzamos j mientras sea posible para encontrar el "mejor match"
        # pero que aún esté dentro del caliper.
        
        min_dist = float("inf")
        chosen_j = None
        
        # Recorremos a partir de j, ya que los anteriores han sido evaluados o usados
        for k in range(j, len(larger_group)):
            score_k = larger_group.loc[k, "propensity_score"]
            
            # Distancia entre scores
            dist = abs(score_i - score_k)
            
            # Si está dentro del caliper y es la más pequeña encontrada hasta ahora
            if dist <= caliper and dist < min_dist:
                min_dist = dist
                chosen_j = k
            
            # Si el score_k se vuelve mayor que score_i + caliper, podemos romper
            # (dado que el larger_group está ordenado, no habrá más matches).
            if score_k > score_i + caliper:
                break
        
        # Si found match (chosen_j no es None)
        if chosen_j is not None:
            # Agregamos ambos a la lista de emparejados
            matched_smaller.append(smaller_group.iloc[i])
            matched_larger.append(larger_group.iloc[chosen_j])
            
            # Eliminamos ese índice del grupo mayor para que no sea reutilizado
            larger_group.drop(chosen_j, inplace=True)
            # Reseteamos el índice del larger_group y lo volvemos a ordenar por score
            larger_group = larger_group.sort_values(by="propensity_score").reset_index(drop=True)
            
            # No avanzamos j de forma secuencial fija, porque reordenamos y resetamos índice
            # de larger_group en cada match. Lo que sí podríamos hacer es mantener la
            # posición 'chosen_j' pero ya el reindexing lo cambia. Por simplicidad,
            # cada vez lo recorremos completo. (Si quisiéramos optimizar más, 
            # podríamos diseñar una búsqueda binaria o uso de nearest neighbors).
            
        # Si no se encontró match, descartamos ese individuo de smaller_group (no hacemos nada).
        # Se "pierde" y no se añade a matched_smaller.

    # 6. Reconstruimos DataFrame final de matches
    matched_smaller = pd.DataFrame(matched_smaller)
    matched_larger = pd.DataFrame(matched_larger)
    
    # Unimos la data en el mismo orden
    matched_data = pd.concat([matched_smaller, matched_larger], axis=0)
    
    # 7. Restituimos el orden original de los datos, si se desea
    matched_data = matched_data.sort_values("propensity_score").reset_index(drop=True)
    
    # (Opcional) Si se quiere reetiquetar el grupo mayor y menor en caso de que
    # se haya intercambiado, se puede realizar aquí. Sin embargo, mientras las
    # columnas de 'treatment_col' sigan correctas, no es necesario.

    return matched_data