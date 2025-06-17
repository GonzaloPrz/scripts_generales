import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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

def estimate_three_way_propensity_scores(df, treatment_col, covariate_cols):
    """
    Estimate propensity scores for a three-level treatment variable using
    multinomial logistic regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data containing the treatment_col and covariate_cols.
    treatment_col : str
        Name of the column indicating the treatment group (0, 1, or 2).
    covariate_cols : list
        List of column names for covariates to be used in the model.
    
    Returns
    -------
    np.ndarray
        An N x 3 array of predicted probabilities, one column per treatment level.
    """
    model = HistGradientBoostingClassifier(learning_rate=1e-3,max_iter=1000)
    model.fit(df[covariate_cols], df[treatment_col])
    # predict_proba returns an N x 3 array of probabilities (for 3 classes)
    propensity_scores = model.predict_proba(df[covariate_cols])
    return propensity_scores

def perform_three_way_matching(
    df, 
    treatment_col, 
    covariate_cols, 
    factor_vars=None, 
    treatment_values=(0, 1, 2), 
    caliper=0.05
):
    """
    Perform 1:1:1 three-way matching based on the Euclidean distance in 
    propensity-score space. Each subject in group A is matched to a single subject 
    in group B and a single subject in group C, provided the distance (in 3D 
    probability space) does not exceed `caliper`.

    Parameters
    ----------
    df : pd.DataFrame
        The data containing the treatment_col and covariate_cols.
    treatment_col : str
        The name of the treatment variable with three categories 
        (e.g., 0, 1, 2).
    covariate_cols : list
        List of covariate column names.
    factor_vars : list, optional
        List of categorical columns that need to be label-encoded for 
        LogisticRegression. Defaults to None.
    treatment_values : tuple, optional
        The unique values of the treatment groups. Defaults to (0, 1, 2).
    caliper : float, optional
        Maximum allowed Euclidean distance in the (p0, p1, p2) space 
        for a valid match. Defaults to 0.05.

    Returns
    -------
    matched_data : pd.DataFrame
        A DataFrame containing only the matched subjects. It will contain 
        three times as many rows as the number of matched triplets.
    """

    if factor_vars is None:
        factor_vars = []

    # 1. Encode categorical columns (if needed)
    for col in factor_vars:
        df[col] = LabelEncoder().fit_transform(df[col])

    # 2. Estimate multinomial propensity scores (p0, p1, p2)
    propensities = estimate_three_way_propensity_scores(df, treatment_col, covariate_cols)
    df["p0"] = propensities[:, 0]
    df["p1"] = propensities[:, 1]
    df["p2"] = propensities[:, 2]

    # 3. Split the DataFrame into the three groups
    groups = {}
    for val in treatment_values:
        group_df = df[df[treatment_col] == val].copy()
        group_df = group_df.sort_values(by=["p0", "p1", "p2"]).reset_index(drop=True)
        groups[val] = group_df

    # Identify the smallest group so we iterate over it first (for efficiency).
    # We'll try to find matches from the other two groups for each subject in 
    # the smallest group.
    group_sizes = {val: len(groups[val]) for val in treatment_values}
    smallest_group_val = min(group_sizes, key=group_sizes.get)
    other_vals = [v for v in treatment_values if v != smallest_group_val]

    matched_smallest = []
    matched_others_1 = []
    matched_others_2 = []

    # 4. Match each subject in the smallest group to one subject in each of 
    #    the other two groups, controlling for a caliper in 3D space of (p0, p1, p2).
    #    We'll do a naive nearest-neighbor approach for each group.
    group_smallest = groups[smallest_group_val]

    # For convenience, define references to the other groups
    group_other_1 = groups[other_vals[0]]
    group_other_2 = groups[other_vals[1]]

    # We'll keep track of unmatched indices from other groups
    # so we can remove them when matched once (no replacement).
    unmatched_idx_1 = set(group_other_1.index)
    unmatched_idx_2 = set(group_other_2.index)

    for i in range(len(group_smallest)):
        print(f'Matching subject {i}/{len(group_smallest)}')
        ps_small = group_smallest.loc[i, ["p0", "p1", "p2"]].values

        # We'll search for the best match in the other two groups 
        # that also meets the caliper requirement.
        min_dist_sum = float("inf")
        chosen_j = None
        chosen_k = None

        for j in unmatched_idx_1:
            ps_1 = group_other_1.loc[j, ["p0", "p1", "p2"]].values
            dist_1 = np.linalg.norm(ps_small - ps_1)  # Euclidean distance

            if dist_1 > caliper:
                # If group_other_1 is sorted, we could break early 
                # but let's just check the entire list for simplicity.
                continue

            for k in unmatched_idx_2:
                ps_2 = group_other_2.loc[k, ["p0", "p1", "p2"]].values
                dist_2 = np.linalg.norm(ps_small - ps_2)

                if dist_2 > caliper:
                    continue

                # total distance = dist_1 + dist_2 
                # (One could also consider max of dist_1 & dist_2, 
                #  or some other weighting scheme.)
                dist_sum = dist_1 + dist_2

                if dist_sum < min_dist_sum:
                    min_dist_sum = dist_sum
                    chosen_j = j
                    chosen_k = k

        # If we found a suitable match
        if chosen_j is not None and chosen_k is not None:
            # Record the matched subjects
            matched_smallest.append(group_smallest.iloc[i])
            matched_others_1.append(group_other_1.loc[chosen_j])
            matched_others_2.append(group_other_2.loc[chosen_k])

            # Remove them from future matching
            unmatched_idx_1.remove(chosen_j)
            unmatched_idx_2.remove(chosen_k)
        # else, that subject remains unmatched (we skip it)

    # 5. Combine matched groups into a single DataFrame
    matched_smallest_df = pd.DataFrame(matched_smallest)
    matched_others_1_df = pd.DataFrame(matched_others_1)
    matched_others_2_df = pd.DataFrame(matched_others_2)

    matched_data = pd.concat([matched_smallest_df, matched_others_1_df, matched_others_2_df], axis=0)
    matched_data = matched_data.sort_values(["p0", "p1", "p2"]).reset_index(drop=True)

    return matched_data