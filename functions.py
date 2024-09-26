import pandas as pd
import numpy as np
from streamlit import warning
import streamlit as st
import datetime
from os import path
from json import load, dump
from io import BytesIO



# def correction_dates_integrale(df_raw, col, date_format='%m/%d/%Y'):
    
#     df = df_raw[[col]].copy()

#     # Convertir toutes les valeurs en chaînes de caractères
#     df[col] = df[col].astype(str)

#     # Remplacer les valeurs spécifiques par des dates de référence
#     df[col] = df[col].replace('2958465', '2099-12-31')

#     # Identifier et convertir les valeurs numériques en dates
#     numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
#     df.loc[numeric_mask, col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[numeric_mask, col].astype(int), 'D')
#     df.loc[numeric_mask, col] = df.loc[numeric_mask, col].astype(str)
    
#     # Supprimer les chaînes de temps
#     df[col] = df[col].str.replace(" 00:00:00", "")
    
#     # Remplacer '9999' et '2999' par '2099'
#     df[col] = df[col].str.replace('9999', '2099')
#     df[col] = df[col].str.replace('2999', '2099')
    
#     # Identifier et convertir les dates au format avec tirets
#     dash_mask = df[col].str.contains('-')
#     slash_mask = df[col].str.contains('/')

#     # Traiter les dates avec tirets
#     if dash_mask.any():
#         df.loc[dash_mask, col] = pd.to_datetime(df.loc[dash_mask, col])

#     # Traiter les dates avec barres obliques
#     if slash_mask.any():
#         # Utiliser to_datetime avec coerce pour tenter les formats les plus courants
#         try:
#             df.loc[slash_mask, col] = pd.to_datetime(df.loc[slash_mask, col], format=date_format)
#         except ValueError:
#             df.loc[slash_mask, col] = pd.to_datetime(df.loc[slash_mask, col], format='%d/%m/%Y')

#     return pd.to_datetime(df[col])

def is_valid_siren(column):
    column = np.asarray(column, dtype=str)
    results = np.zeros_like(column, dtype=bool)

    for i, siren in np.ndenumerate(column):
        if len(siren) != 9:
            results[i] = False
        else:
            siren_digits = [digit for digit in siren if digit.isdigit()]
            if len(siren_digits) != 9:
                results[i] = False
            else:
                siren_digits = [int(digit) for digit in siren_digits]
                siren_digits.reverse()
                total = 0
                for j, digit in enumerate(siren_digits):
                    if j % 2 == 0:
                        total += digit
                    else:
                        doubled_digit = digit * 2
                        total += doubled_digit if doubled_digit < 10 else (doubled_digit - 9)
                results[i] = total % 10 == 0
    return results

def is_valid_siret(column):
    column = np.asarray(column, dtype=str)
    results = np.zeros_like(column, dtype=bool)

    for i, siret in np.ndenumerate(column):
        if len(siret) != 14:
            results[i] = False
        else:
            siret_digits = [digit for digit in siret if digit.isdigit()]
            if len(siret_digits) != 14:
                results[i] = False
            else:
                siren_part = ''.join(siret_digits[:9])  # First 9 digits form the SIREN
                if not is_valid_siren([siren_part])[0]:  # Pass SIREN as a string
                    results[i] = False
                else:
                    siret_digits = [int(digit) for digit in siret_digits]
                    siret_digits.reverse()
                    total = 0
                    for j, digit in enumerate(siret_digits):
                        if j % 2 == 0:
                            total += digit
                        else:
                            doubled_digit = digit * 2
                            total += doubled_digit if doubled_digit < 10 else (doubled_digit - 9)
                    results[i] = total % 10 == 0
    return results

def check_validity_siren_siret(df_raw, col_name, id_type='siren', skip_na=False, return_df=False, warn_type='warning'):
    if id_type not in ['siren', 'siret']:
        raise ValueError("id_type must be either 'siren' or 'siret'")
    
    df = df_raw[[col_name]]

    # Appliquer la validation en fonction du type d'identifiant
    if id_type == 'siren':
        df['is_valid'] = is_valid_siren(df[col_name])
        invalid_ids = df[~df['is_valid']]
        message = f"{len(invalid_ids)} SIRENs ({(len(invalid_ids)/len(df))*100:.2f}%) dans {col_name} sont invalides."
    elif id_type == 'siret':
        df['is_valid'] = is_valid_siret(df[col_name])
        invalid_ids = df[~df['is_valid']]
        message = f"{len(invalid_ids)} SIRETs ({(len(invalid_ids)/len(df))*100:.2f}%) dans {col_name} sont invalides."
    
    if not invalid_ids.empty:
        if return_df:
            return df_raw.loc[invalid_ids.index]
        return {warn_type: message}
    
    return None


#________________________________________________________ Controle Functions_____________________________________________________________

def store_result(col, func, result_dict):
    """
    This function takes a function and a dictionary as input, and stores th rusults of the function in the provided dictionary
    """
    
    result = func
    if result:
        if col not in result_dict:
                result_dict[col] = []
        result_dict[col].append(result)
    
def rename_func(df, type_fichier, type_bdd, assureur, rename_dict=None, keep_default_dict=True, warn=True, update_json=True, json_file=r'C:\Users\Yacine AMMI\Yacine\Notebooks\AOPS\Scripts\renaming.json'):
    """
    Rename columns in a DataFrame based on a rename dictionary, and optionally update a JSON file with the renaming rules.

    Args:
    df (pd.DataFrame): The DataFrame to rename.
    type_fichier (str): The type of file (e.g., "santé").
    type_bdd (str): The type of database (e.g., "prestations", "cotisations", "effectifs").
    assureur (str): The insurance provider (e.g., "aesio").
    rename_dict (dict): Optional dictionary with renaming rules.
    keep_default_dict (bool): Whether to merge the existing dictionary with the new one.
    warn (bool): Whether to warn about columns that were not renamed.
    update_json (bool): Whether to update the JSON file with the new renaming rules.
    json_file (str): Path to the JSON file to update.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    type_bdd = type_bdd.lower()
    assureur = assureur.lower()
    
    if type_bdd not in ['prestations', 'cotisations', 'effectifs']:
        raise ValueError('type_bdd doit être "prestations", "cotisations" ou "effectifs".')
    
    # Load existing rename dictionary from JSON file
    if path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = load(file)
    else:
        print('JSON file not found, creating a new one.')
        data = {}

    # Initialize the nested structure if missing
    if type_fichier not in data:
        data[type_fichier] = {}
    
    if type_bdd not in data[type_fichier]:
        data[type_fichier][type_bdd] = {}
    
    if assureur not in data[type_fichier][type_bdd]:
        data[type_fichier][type_bdd][assureur] = {}

    # Get the existing dictionary for renaming
    existing_dict = data[type_fichier][type_bdd][assureur]

    # Handle the keep_default_dict logic
    if rename_dict is not None:
        if keep_default_dict:
            # Merge the existing dictionary with the new renaming rules
            existing_dict.update(rename_dict)
        else:
            # Replace the existing dictionary entirely with the new one
            existing_dict = rename_dict

    # Update the nested dictionary in `data`
    data[type_fichier][type_bdd][assureur] = existing_dict

    # Save updated dictionary back to JSON file
    if update_json:
        with open(json_file, 'w', encoding='utf-8') as file:
            dump(data, file, ensure_ascii=False, indent=4)

    # Rename columns in the DataFrame
    renamed_df = df.rename(columns=existing_dict)

    # Check for columns that were not renamed
    renamed_columns = set(existing_dict.keys())
    existing_columns = set(df.columns)
    not_renamed = existing_columns - renamed_columns

    if not_renamed and warn:
        print("Warning: The following columns were not renamed:", not_renamed)

    return renamed_df


# def rename_func(df, type_fichier, type_bdd, assureur, rename_dict=None, keep_default_dict=True, warn=True, update_json=True, json_file=r'C:\Users\Yacine AMMI\Yacine\Notebooks\AOPS\Scripts\renaming.json'):
    
#     type_bdd = type_bdd.lower()
#     assureur = assureur.lower()
    
#     if type_bdd not in ['prestations', 'cotisations', 'effectifs']:
#         raise ValueError('type_bdd doit etre prestations, cotisations ou effectifs')
    
#     # if assureur not in ['ag2r', 'macif', 'aesio']:
#     #     raise ValueError('assureur doit etre ag2r, macif ou aesio')
    
#     # Load existing rename dictionary from JSON file
#     if path.exists(json_file):
#         with open(json_file, 'r', encoding='utf-8') as file:
#             data = load(file)
#             existing_dict = data.get(type_fichier, {}).get(type_bdd, {}).get(assureur, {})
#     else:
#         print('JSON file not found, creating a new one.')
#         data = {}
#         existing_dict = {}
        
#     # Update existing dictionary with new values
#     if rename_dict is not None:
#         if keep_default_dict:
#             existing_dict.update(rename_dict)
#         else:
#             existing_dict = rename_dict
        
#     if type_bdd not in data:
#         data[type_bdd] = {}
#     data[type_bdd][assureur] = existing_dict

#     # Save updated dictionary back to JSON file
#     if update_json:
#         with open(json_file, 'w', encoding='utf-8') as file:
#             dump(data, file, ensure_ascii=False, indent=4)

#     # Rename columns in the DataFrame
#     renamed_df = df.rename(columns=existing_dict)

#     # Check for columns that were not renamed
#     renamed_columns = set(existing_dict.keys())
#     existing_columns = set(df.columns)
#     not_renamed = existing_columns - renamed_columns

#     if not_renamed and warn:
#         print("Warning: The following columns were not renamed:", not_renamed)

#     return renamed_df

def check_mandatory_cols(df, cols, df_name=None, raise_err=True):
    
    """Check if all mandatory columns are present in the DataFrame.
        raises an error if """
    
    missing_cols = set(cols) - set(df.columns)
    
    if missing_cols:
        if raise_err:
            if df_name:
                raise ValueError(f'The following mandatory columns are missing from {df_name}: {missing_cols}')
            else:
                raise ValueError(f"The following mandatory columns are missing: {missing_cols}")
        else:
            if df_name:
                print(f"Warning: The following mandatory columns are missing from {df_name}: {missing_cols}, please add them to rename dict")
            else:
                print(f"Warning: The following mandatory columns are missing: {missing_cols}, please add them to rename dict")
                
    return missing_cols

def check_missing(df, col, warn_type='warning'):
    """Check for missing values in a column."""
    # Check for missing values in a column
    missing = df[col].isnull().sum()
    if missing > 0:
        return {warn_type: f'{missing} lignes ({(missing/len(df))*100 :.2f}%) ont des valeurs manquantes dans {col}.'}
    return None

def check_id_length(df, col_name, length, skip_na=False, return_df=False, warn_type='warning'):
    if not skip_na:
        invalid_ids = df[df[col_name].str.len() != length]
    else:
        invalid_ids = df[df[col_name].notnull() & (df[col_name].str.len() != length)]
        
    if not invalid_ids.empty:
        if return_df:
            return invalid_ids
        return {warn_type: f'{len(invalid_ids)} IDs ({(len(invalid_ids)/len(df))*100:.2f})% en {col_name} ne contiennent pas {length} caractères.'}
    return None

def check_ids_presence(df_t1, df_t2, col_name, return_df=False, warn_type='warning'):
    missing_ids = set(df_t2[col_name])-set(df_t1[col_name])
    if missing_ids:
        if return_df:
            return missing_ids
        percentage = len(missing_ids) / len(set(df_t2[col_name])) * 100
        return {warn_type : f'{len(missing_ids)} IDs de {col_name}  de t ne sont pas présents dans t-1, ce qui représente {percentage:.2f}% du total des identifiants de t.'}
    return None

def check_special_chars(df, col_name, special_chars, return_df=False, warn_type='warning'):
    # Créer un modèle regex pour les caractères spéciaux
    pattern = f"[{''.join(special_chars)}]"
    
    # Appliquer str.contains sur la colonne remplie
    special_ids = df[col_name].str.contains(pattern, regex=True)

    # Filtrer les identifiants qui contiennent des caractères spéciaux
    special_ids = df[special_ids.fillna(False)]
    
    if not special_ids.empty:
        if return_df:
            return special_ids
        return {warn_type : f'{len(special_ids)} lignes ou les IDs dans {col_name} contiennent des caractères spéciaux {special_chars}.'}
    return None

def check_id_length_and_content(df, col_name, length, required_chars, return_df=False, warn_type='warning'):
    invalid_ids = df[df[col_name].notnull() & ((df[col_name].str.len() != length) | ~df[col_name].str.contains('|'.join(required_chars)))]
    if not invalid_ids.empty:
        if return_df:
            return invalid_ids
        return {warn_type: f'{len(invalid_ids)} IDs dans {col_name} ne contient pas {length} caractères ou ne contient pas {required_chars}.'}
    return None

def check_date_formats(df_raw, col, date_patterns=None, return_df=False, warn_type='warning'):
    if date_patterns is None:
        date_patterns = [
            "%Y-%m-%d",  # "YYYY-MM-DD"
            "%d/%m/%Y",  # "DD/MM/YYYY"
            "%d/%m/%y",  # "D/M/YYYY"
            "%m/%d/%Y",  # "MM/DD/YYYY"
            "%m/%d/%Y",  # "M/D/YYYY"
            "%Y/%m/%d",  # "YYYY/MM/DD"
            "%m/%d/%y",  # "M/D/YY"
            "%m/%d/%y",  # "MM/DD/YY"
            "%d/%m/%y",  # "DD/MM/YY"
            "%d/%m/%y",  # "D/M/YY"
            "%Y"         # "YYYY"
        ]

    # Convertir toutes les valeurs en chaînes de caractères
    df = df_raw[[col]].copy()
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(" 00:00:00", "")
    df[col] = df[col].str.replace("2999", "2099")
    
    # Initialiser un masque pour les dates valides
    valid_mask = pd.Series([False] * len(df), index=df.index)
    
    # Vérifier les formats spécifiés
    for pattern in date_patterns:
        try:
            valid_dates = pd.to_datetime(df[col], format=pattern, errors='coerce').notna()
            valid_mask |= valid_dates
        except Exception :
            continue

    # Calculer le nombre de dates invalides
    invalid_dates_count = (~valid_mask).sum()
    
    if invalid_dates_count > 0:
        if return_df:
            return df[~valid_mask]
        percentage=(invalid_dates_count/df.shape[0])*100
        return {warn_type: f'{invalid_dates_count} lignes ({percentage:.2f}%) dans {col} ont des formats de date non valides.'}
    
    return None

def correction_dates_integrale(df_raw, col, date_formats=None, add_format=None):
    
    df = df_raw[[col]].copy()
    # Convertir toutes les valeurs en chaînes de caractères
    df[col] = df[col].astype(str)
    # Remplacer les valeurs spécifiques par des dates de référence
    df[col] = df[col].replace('2958465', '2099-12-31')
    # Identifier et convertir les valeurs numériques en dates
    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
    df.loc[numeric_mask, col] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df.loc[numeric_mask, col].astype(int), 'D')
    df.loc[numeric_mask, col] = df.loc[numeric_mask, col].astype(str)
    # Supprimer les chaînes de temps
    df[col] = df[col].str.replace(" 00:00:00", "")
    # Remplacer '9999' et '2999' par '2099'
    df[col] = df[col].str.replace('9999', '2099')
    df[col] = df[col].str.replace('2999', '2099')
    # Identifier et convertir les dates au format avec tirets
    dash_mask = df[col].str.contains('-')
    slash_mask = df[col].str.contains('/')
    # Traiter les dates avec tirets
    if dash_mask.any():
        df.loc[dash_mask, col] = pd.to_datetime(df.loc[dash_mask, col])
    # Traiter les dates avec barres obliques
    if slash_mask.any():
        if not date_formats:
            # Utiliser to_datetime avec coerce pour tenter les formats les plus courants
            date_formats = [
                "%Y-%m-%d",  # "YYYY-MM-DD"
                "%d/%m/%Y",  # "DD/MM/YYYY"
                "%d/%m/%y",  # "D/M/YYYY"
                "%d/%m/%Y",  # "DD/M/YYYY"
                "%m/%d/%Y",  # "MM/DD/YYYY"
                "%m/%d/%Y",  # "M/D/YYYY"
                "%Y/%m/%d",  # "YYYY/MM/DD"
                "%m/%d/%y",  # "M/D/YY"
                "%m/%d/%y",  # "MM/DD/YY"
                "%d/%m/%y",  # "DD/MM/YY"
                "%d/%m/%y",  # "D/M/YY"
            ]
            
        # Convertir la col "dates" en objets datetime
        for format in date_formats:
            try:
                df.loc[slash_mask, col] = pd.to_datetime(df.loc[slash_mask, col], format=format)
                break
            except ValueError:
                continue
    return pd.to_datetime(df[col])

def check_advanced_date_formats(df, col, date_formats=None, warn_type='warning'):
    try:
        if not date_formats:
            date_formats = [
            "%Y-%m-%d",  # "YYYY-MM-DD"
            "%d/%m/%Y",  # "DD/MM/YYYY"
            "%d/%m/%y",  # "D/M/YYYY"
            "%d/%m/%Y",  # "DD/M/YYYY"
            "%m/%d/%Y",  # "MM/DD/YYYY"
            "%m/%d/%Y",  # "M/D/YYYY"
            "%Y/%m/%d",  # "YYYY/MM/DD"
            "%m/%d/%y",  # "M/D/YY"
            "%m/%d/%y",  # "MM/DD/YY"
            "%d/%m/%y",  # "DD/MM/YY"
            "%d/%m/%y",  # "D/M/YY"
        ]
        date_col = correction_dates_integrale(df, col, date_formats=date_formats)
        return None
    except (ValueError):
        return {warn_type: f'Les lignes de {col} ont des formats de date non valides.'}

def compare_dates(df_raw, col1, col2, condition='<', return_df=False, warn_type='warning'):
    # Crée une copie des colonnes d'intérêt
    df = df_raw[[col1, col2]].copy()
    
    # Corrige les dates
    df[col1] = correction_dates_integrale(df, col1)
    df[col2] = correction_dates_integrale(df, col2)
    
    # Utilise query pour vérifier les conditions
    condition_map = {
        '<': f"{col1} < {col2}",
        '>': f"{col1} > {col2}",
        '=': f"{col1} == {col2}",
        '!=': f"{col1} != {col2}",
        '<=': f"{col1} <= {col2}",
        '>=': f"{col1} >= {col2}"
    }
    
    
    
    if condition not in condition_map:
        raise ValueError(f"Condition '{condition}' is not supported. Use one of: '<', '>', '=', '!=', '<=', '>='.")
    
    query_str = condition_map[condition]
    valid_rows = df.query(query_str)
    invalid_rows = df.drop(valid_rows.index)
    # drop na vaalues from invalid_rows
    invalid_rows.dropna(subset=[col1, col2], inplace=True)
    
    # Calcule le nombre et le pourcentage de lignes qui ne respectent pas la condition
    num_invalid = len(invalid_rows)
    total_rows = len(df)
    percentage_invalid = (num_invalid / total_rows) * 100
    
    # Créer le message d'avertissement ou d'alerte
    if num_invalid > 0:
        message = {warn_type: f'{num_invalid} lignes ({percentage_invalid:.2f}%) ne remplissent pas la condition {col1} {condition} {col2}.'}
    else:
        message = None
    
    # Retourner le DataFrame des lignes invalides si return_df est True
    if return_df:
        return df_raw.loc[invalid_rows.index]
    
    return message

def check_valid_values(df, col, valid_values, strip=False, replace=False, lower=False, warn_type='warning'):
    """
    Check if values in a column are valid.
    
    Args:
        df (_type_): _description_
        col (_type_): _description_
        valid_values (_type_): _description_
        strip (bool, optional): _description_. Defaults to False.
        replace (bool, optional): _description_. Defaults to False.
        lower (bool, optional): _description_. Defaults to False.
        warn_type (str, optional): _description_. Defaults to 'warning'.

    Returns:
        _type_: _description_
    """
    mod_col = df[col].copy()
    if strip:
        mod_col = mod_col.str.strip()
        if lower:
            mod_col = mod_col.str.lower()
            if replace:
                mod_col = mod_col.str.replace(' ', '')
    
    invalid_values = df.loc[~mod_col.isin(valid_values), col].unique()
        
    if len(invalid_values) > 0:
        if len(invalid_values) > 5:
            invalid_values = invalid_values[:5]
        return {warn_type: f'{len(invalid_values)} lignes de {col} ont des valeurs non valides {invalid_values}.'}
    return None

def restructure_results(results):
    restructured = {'warnings': {}, 'alerts': {}}
    
    for key, value in results.items():
        if not value:
            continue
        
        for entry in value:
            if 'warning' in entry:
                if key not in restructured['warnings']:
                    restructured['warnings'][key] = []
                restructured['warnings'][key].append(entry['warning'])
            elif 'alert' in entry:
                if key not in restructured['alerts']:
                    restructured['alerts'][key] = []
                restructured['alerts'][key].append(entry['alert'])
    
    return restructured

def check_date_range(df_raw, col, min_date, max_date, return_df=False, warn_type='alert'):
    """Check if the date range is within the expected range, if not, return number and %"""
    # Crée une copie de la colonne d'intérêt
    df = df_raw[[col]].copy()
    
    # Corrige les dates
    df[col] = correction_dates_integrale(df, col)
    
    min_date, max_date = pd.Timestamp(min_date), pd.Timestamp(max_date)
    
    # Filtrer les dates invalides
    invalid_dates = df[~df[col].between(min_date, max_date, inclusive="both")]
    
    # Calculer le nombre et le pourcentage de dates invalides
    num_invalid = len(invalid_dates)
    total_rows = len(df)
    percentage_invalid = (num_invalid / total_rows) * 100
    
    # Créer le message d'alerte ou d'avertissement
    if num_invalid > 0:
        message = {warn_type: f"{num_invalid} les lignes ({percentage_invalid :.2f}%) dans {col} ne sont pas comprises dans l'intervalle de dates {min_date} à {max_date}."}
    else:
        message = None
    
    # Retourner le DataFrame des lignes invalides si return_df est True
    if return_df:
        return message, df_raw.loc[invalid_dates.index]
    
    return message
    
        

### -----------------------------prest---------------------------------

def check_taux_ss(df_raw, col, return_col_corr=False, return_df=False, warn_type='warning'):
    """Checker / corriger les valeurs de taux_ss dans la colonne spécifiée et retourner les messages d'avertissement."""
    
    warnings = ''
    
    # Créer une copie pour conserver l'originalité des données
    df = df_raw[[col]].copy()
    df[col] = df[col].astype(str).str.replace(',', '.').str.strip()

    # Traiter les pourcentages
    mask_percent = df[col].str.endswith('%')
    percent_count = mask_percent.sum()
    df.loc[mask_percent, col] = df.loc[mask_percent, col].str.rstrip('%').astype(float)

    # Convertir en float les valeurs non pourcentages
    mask_non_percent = ~mask_percent
    df.loc[mask_non_percent, col] = pd.to_numeric(df.loc[mask_non_percent, col], errors='coerce')

    if percent_count > 0:
        warnings += f"{percent_count} valeurs de  ({(percent_count/len(df))*100 :.2f})% dans {col} contiennent '%'."

    df[col] = df[col].astype(float)
    if return_col_corr:
        return df[col]
    
    outside_range_T_SS = df[ (df[col] < 0) | (df[col] > 1) ]
    negatives = df[(df[col] < 0)]
    
    if warnings != '':
        warnings = f'{warnings} \n'

    if len(outside_range_T_SS) > 0:
        
        warnings += f"""{outside_range_T_SS.shape[0] :,.0f} lignes avec des valeurs en dehors de l'intervalle [0, 1] (où {negatives.shape[0] :,.0f} sont des négatives)"""
        
    if warnings != '':
        if return_df:
            return df_raw[mask_percent.index.union(outside_range_T_SS.index)]
        return {warn_type: warnings}
    else:
        return None
    

def check_fr_base_taux(df_raw, fr_col='FR', base_ss_col='Base_SS', taux_ss_col='Taux_SS', warn_type='warning'):
    df = df_raw[[fr_col, base_ss_col, taux_ss_col]]
    df[taux_ss_col] = check_taux_ss(df, taux_ss_col, return_col_corr=True)
    
    condition = (df[fr_col].isna()) | (df[fr_col] == 0)
    invalid_rows = df[condition & (~(df[base_ss_col].isna() | (df[base_ss_col] == 0)) | ~(df[taux_ss_col].isna() | (df[taux_ss_col] == 0)))]
    num_invalid = len(invalid_rows)
    total_rows = len(df)
    percentage_invalid = (num_invalid / total_rows) * 100
    
    if num_invalid > 0:
        return {warn_type: f'{num_invalid} lignes ({percentage_invalid:.2f}%) ne remplissent pas la condition selon laquelle lorsque {fr_col} est vide ou 0, {base_ss_col} et {taux_ss_col} doivent être vides ou 0.'}
    return None

def check_r_ss_equality(df_raw, r_ss_col='R_SS', base_ss_col='Base_SS', taux_ss_col='Taux_SS', tolerance=1, return_df=False, warn_type='warning'):
    
    df = df_raw[[r_ss_col, base_ss_col, taux_ss_col]]
    df[taux_ss_col] = check_taux_ss(df, taux_ss_col, return_col_corr=True)
    
    incorrect_r_ss = df[np.abs(df[r_ss_col] - (df[taux_ss_col] * df[base_ss_col])) > tolerance]
    num_incorrect = len(incorrect_r_ss)
    total_rows = len(df)
    percentage_incorrect = (num_incorrect / total_rows) * 100
    
    if num_incorrect > 0:
        if return_df:
            return df_raw[incorrect_r_ss.index]
        else:
            return {warn_type: f'{num_incorrect} lignes ({percentage_incorrect : .2f}%) où {r_ss_col} != {base_ss_col} * {taux_ss_col} (différence > {tolerance}).'}
    return None

def check_r_ss_without_base_taux(df_raw, r_ss_col='R_SS', base_ss_col='Base_SS', taux_ss_col='Taux_SS', return_df=False, warn_type='warning'):
    
    df = df_raw[[r_ss_col, base_ss_col, taux_ss_col]]
    df[taux_ss_col] = check_taux_ss(df, taux_ss_col, return_col_corr=True)
    
    
    outside_range_r_ss = df[
        (((df[taux_ss_col] == 0) | (df[taux_ss_col].isna())) & ((df[base_ss_col] == 0) | (df[base_ss_col].isna()))) &
        ((df[r_ss_col] != 0) & (df[r_ss_col].notna()))
    ]
    num_outside_range = len(outside_range_r_ss)
    total_rows = len(df)
    percentage_outside_range = (num_outside_range / total_rows) * 100
    
    
    if num_outside_range > 0:
        if return_df:
            return df_raw[outside_range_r_ss.index]
        else:
            return {warn_type: f'{num_outside_range} lignes ({percentage_outside_range:.2f}%) où {r_ss_col} est sans {base_ss_col} ou {taux_ss_col}.'}
    return None

def check_rac_negative(df, col='RàC', warn_type='warning'):
    rac_negative = df[df[col] < 0]
    num_negative = len(rac_negative)
    total_rows = len(df)
    percentage_negative = (num_negative / total_rows) * 100
    
    if num_negative > 0:
        return {warn_type: f'{num_negative} lignes ({percentage_negative:.2f}%) dans {col} ont des valeurs négatives.'}
    return None

def check_quantite_acte(df, col='quantité_acte', warn_type='warning'):
    warnings = []

    # Vérification des décimaux
    decimaux = df[df[col] % 1 != 0]
    
    df_len = len(df)
    
    if len(decimaux) > 0:
        warnings.append(f'{len(decimaux)} ({(len(decimaux)/df_len)*100:.2f})% lignes avec des valeurs décimales (non entières) dans {col}.')

    # Vérification des valeurs négatives
    negatives = df[df[col] < 0]
    if len(negatives) > 0:
        warnings.append(f'{len(negatives)} ({(len(negatives)/df_len)*100:.2f})% lignes dont les valeurs sont négatives dans {col}.')

    # Vérification des valeurs nulles
    nulles = df[df[col] == 0]
    if len(nulles) > 0:
        warnings.append(f'{len(nulles)} ({(len(nulles)/df_len)*100:.2f})% lignes avec des valeurs nulles (== 0) dans {col}.')

    if warnings:
        return {warn_type: ' | '.join(warnings)}
    return None

#--------------------------- Cotisations---------------------------
def check_trimester_year(df_raw, col, current_year, tollerance=1, warn_type='warning'):
    """Verifie si tr_surv n'excede pas l'année t+1."""
    
    df = df_raw[[col]].copy()
    
    patterns = [
        r'^\d{4}',         # "YYYYt(1-4)", "YYYYT(1-4)", "YYYY(1-4)"
        r'\d{4}$'          # "t(1-4)YYYY", "T(1-4)YYYY"
    ]
    
    uniques = df[col].unique()
    
    
            
    return None

def check_trimester_format(df, col='tr_surv', patterns=None, warn_type='warning'):
    if patterns is None:
        patterns = [
            r'^\d{4}t[1-4]$',   # "YYYYt(1-4)"
            r'^\d{4}T[1-4]$',   # "YYYYT(1-4)"
            r'^t[1-4]\d{4}$',   # "t(1-4)YYYY"
            r'^T[1-4]\d{4}$',   # "T(1-4)YYYY"
            r'^\d{4}[1-4]$',    # "YYYY(1-4)"
        ]
    
    df_len = len(df)
    unmatched_rows = df.copy()
    
    for pattern in patterns:
        matches = df[df[col].str.match(pattern, na=False)]
        unmatched_rows = unmatched_rows[~unmatched_rows.index.isin(matches.index)]
        
    unmatched_count = len(unmatched_rows)
    
    if unmatched_count > 0:
        return {warn_type:f'{unmatched_count} ({(unmatched_count/df_len)*100:.2f}%) lignes ne correspondent à aucun des modèles spécifiés dans {col}.'}
    else:
        return None
    
def chek_col_float(df_raw, col, warn_type='alert'):
    """Vérifie si une colonne est composée de nombres."""
    # Vérifie si une colonne est composée de nombres.
    df = df_raw[[col]].copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    len_non_float = df[col].isna().sum()
    if len_non_float > 0:
        return {warn_type:f'{len_non_float} ({(len_non_float/len(df_raw))*100:.2f})% valeurs non numériques dans {col}'}
    else:
        return None
    
def check_siren_siret(df, siren_col='siren', siret_col='siret', return_df=False, warn_type='warning'):
    """Vérifie si les SIREN corespond bien au SIRET."""
    
    # Vérifie si les SIREN corespond bien au SIRET.
    siren_siret = df[[siren_col, siret_col]]
    
    siren_siret = siren_siret[siren_siret[siret_col].str.len() >= 9]
    
    problematic_rows =  siren_siret[siren_siret[siren_col] != siren_siret[siret_col].str[:9]]
    
    if len(problematic_rows) > 0:
        if return_df:
            return problematic_rows
        else:
            return {warn_type:f'{len(problematic_rows)} lignes ont un SIREN qui ne correspond pas au SIRET.'}
        
def remove_duplicate_columns(df):
    """
    Check for duplicate columns in a DataFrame and remove them if found.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with duplicate columns removed.
    """
    # Identify duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    
    if duplicate_columns:
        # Print a warning about the duplicate columns
        print(f"ALERTE!!!!!!: Des colonnes en double ont été trouvées : {duplicate_columns}. Elles seront supprimées.")
        
        # Drop duplicate columns, keeping the first occurrence
        df = df.loc[:, ~df.columns.duplicated()]
    
    return df
#-------------------------------------------- Global check functions ----------------------------------------------------        
        
def check_effectifs(assureur, df_raw, dft_1_raw=None, rename_dict=None, raise_err=False, results_by_type=False):
    
    assureur = assureur.lower()
    if assureur not in ['aesio', 'ag2r', 'macif']:
        raise ValueError('assureur doit etre "prestations", "cotisations" ou "effectifs"')
    
    resultats = {}
    mandatory_cols = ['id_ent', 'id_assuré', 'id_bénéf', 'type_bénéf', 'date_naissance', 'niveau_couverture_oblg', 'niveau_couverture_fac', 'cat_assuré', 'date_adh_cat', 'date_sortie_cat', 'date_adh_bénéf', 'date_sortie_bénéf', 'sexe', 'code_postal', 'régim_ss', 'régime', 'structure_cot']
    
    if assureur == 'aesio':
        mandatory_cols.append('siren')
    
    # Rename data
    df = rename_func(df_raw, type_fichier='santé', type_bdd='effectifs', assureur=assureur, rename_dict=rename_dict)
    if dft_1_raw is not None:
        df_t1 = rename_func(dft_1_raw, type_fichier='santé', type_bdd='effectifs', assureur=assureur, rename_dict=rename_dict)
    
    # Check duplicate columns
    df = remove_duplicate_columns(df)
    
    # check for missing cols
    check_mandatory_cols(df, mandatory_cols, raise_err=raise_err)

    
    ## ==========================Check missing values========================
    # check for missing values
    for col in mandatory_cols:
        if col in df.columns:
            resultats[col] = []
            
            if col in ['siren', 'id_bénéf', 'id_assuré']:
                warn_type='alert'
            elif (col == 'id_ent') & (assureur != 'aesio'):
                warn_type='alert'
            else:
                warn_type='warning'
            
            store_result(col, check_missing(df, col, warn_type=warn_type), resultats)
    
    ##------------------- SIREN---------------------
    if assureur == 'aesio':
        if 'siren' in df.columns:
            
            store_result('siren', check_id_length(df, 'siren', 9, skip_na=True), resultats)

    ##------------------- id_ent---------------------

    
    ##------------------- id_assuré---------------------
    if 'id_assuré' in df.columns:
        if assureur in ['ag2r', 'macif']:
            
            store_result('id_assuré', check_id_length(df, 'id_assuré', length=8, warn_type='warning'), resultats)

        if assureur == 'aesio':
            if dft_1_raw is not None:
                
                store_result('id_assuré', check_ids_presence(df_t1, df, 'id_assuré', warn_type='warning'), resultats)
            
            store_result('id_assuré', check_special_chars(df, 'id_assuré', ['/', '\\', '_', '-'], warn_type='warning'), resultats)

            
    ##------------------- id_bénéf---------------------
    if 'id_bénéf' in df.columns:
        if assureur in ['ag2r', 'macif']:
            
            store_result('id_bénéf', check_id_length_and_content(df, 'id_bénéf', length=16, required_chars=['M', 'F'], warn_type='warning'), resultats)
        
        if assureur == 'aesio':
            if dft_1_raw is not None:
                
                store_result('id_bénéf', check_ids_presence(df_t1, df, 'id_bénéf', warn_type='warning'), resultats)
            
            store_result('id_bénéf', check_special_chars(df, 'id_bénéf', ['/', '\\', '_', '-'], warn_type='warning'), resultats)
                
    ##------------------- date_naissance ou annee_naissance---------------------
    if 'date_naissance' in df.columns:
        
        store_result('date_naissance', check_date_formats(df, 'date_naissance'), resultats)
    
    elif 'annee_naissance' in df.columns:
        resultats['date_naissance'] = []
        store_result('date_naissance', check_missing(df, 'annee_naissance', warn_type='warning'), resultats)
        
        store_result('date_naissance', check_date_formats(df, 'annee_naissance', date_patterns=['%Y']), resultats)
    
    ## ==========================Check of valid values======================== cols:['type_bénéf', 'niveau_couverture_fac', 'cat_assuré']
    valid_values_config={
        'type_bénéf': {'valid_values':['A', 'E', 'C', 'P', 'M', 'Assuré', 'Conjoint', 'Enfant', 'AD', 'CJ', 'EN', 'MP'],'strip':False, 'replace':False, 'lower':False},
        'niveau_couverture_fac': {'valid_values':['base', 'option1', 'option2', 'conf', 'confort', 'plus', 'confort+'],'strip':True, 'replace':True, 'lower':True},
        'cat_assuré': {'valid_values':['actif', 'actifs', 'autre', 'chômeur', 'retraité', 'portabilité', 'loi evin', 'ayants-droits', 'suspendu'],'strip':True, 'replace':False, 'lower':True}
    }
    
    for col, config in valid_values_config.items():
        if col in df.columns:
            
            store_result(col, check_valid_values(df, col, **config, warn_type='warning'), resultats)
                
    ##------------------- Dates affiliations / sorties ---------------------
    for col in ['date_adh_cat', 'date_sortie_cat', 'date_adh_bénéf', 'date_sortie_bénéf']:
        if col in df.columns:
            
            store_result(col, check_advanced_date_formats(df, col, warn_type='warning'), resultats)
                
    if ('date_adh_cat' in df.columns) and ('date_adh_bénéf' in df.columns):
        
        store_result('date_adh_cat', compare_dates(df, 'date_adh_bénéf', 'date_adh_cat', condition='<=', return_df=False, warn_type='warning'), resultats)
                
    if ('date_sortie_cat' in df.columns) and ('date_sortie_bénéf' in df.columns):
        
        store_result('date_sortie_cat', compare_dates(df, 'date_sortie_bénéf', 'date_sortie_cat', condition='>=', return_df=False, warn_type='warning'), resultats)
    
    if results_by_type:
        return restructure_results(resultats)
    else:
        return resultats


def check_prestations(assureur, df_raw, dates, dft_1_raw=None, rename_dict=None, raise_err=False):
    
    assureur = assureur.lower()
    if assureur not in ['aesio', 'ag2r', 'macif']:
        raise ValueError("assureur doit etre 'aesio', 'ag2r' ou 'macif'")
    
    try:
        date_debut, date_fin = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    except Exception:
        raise ValueError("dates doit etre une liste de deux dates au format YYYY-MM-DD")
    
    resultats = {}
    mandatory_cols = ['id_ent', 'id_assuré', 'id_bénéf', 'type_bénéf', 'niveau_couverture_fac', 'type_bénéf', 'cat_assuré', 'regime', 'sexe', 'code_acte', 'libellé_acte', 'famille_acte', 'quantité_acte', 'date_soins', 'date_paiement', 'FR', 'Base_SS', 'Taux_SS', 'R_SS', 'RC_Base', 'RC_Option', 'RC_Autre', 'RàC']
    
    if assureur == 'aesio':
        mandatory_cols.append('siren')
    
    # Rename data
    df = rename_func(df_raw, type_fichier='santé',  type_bdd='prestations', assureur=assureur, rename_dict=rename_dict)
    
    if dft_1_raw is not None:
        dft_1 = rename_func(dft_1_raw, type_fichier='santé', type_bdd='prestations', assureur=assureur, rename_dict=rename_dict)
    
    # Check duplicate columns
    df = remove_duplicate_columns(df)
    
    # check for missing cols
    check_mandatory_cols(df, mandatory_cols, df_name='df1', raise_err=raise_err)
    
    ## ==========================Check missing values========================
    # check for missing values
    for col in mandatory_cols:
        if col in df.columns:
            resultats[col] = []
            
            if col in ['siren', 'id_bénéf', 'id_assuré']:
                warn_type='alert'
            elif (col == 'id_ent') & (assureur != 'aesio'):
                warn_type='alert'
            else:
                warn_type='warning'
            
            store_result(col, check_missing(df, col, warn_type=warn_type), resultats)
    
    ## ==========================Dates========================
    for col in ['date_soins', 'date_paiement']:
        if col in df.columns:
            
            store_result(col, check_date_formats(df, col, warn_type='warning'), resultats)
                
    if 'date_paiement' in df.columns:
        
        store_result('date_paiement', check_date_range(df, 'date_paiement', min_date=date_debut, max_date=date_fin, return_df=False, warn_type='alert'), resultats)
    ## ==========================Check of valid values======================== cols:['famille_acte', 'niveau_couverture_fac', 'cat_assuré']
    valid_values_config={
        'famille_acte': {'valid_values':["1. hospitalisation", "2. consultations et visites", "3. autres soins courants", "4. pharmacie", "5. optique", "6. dentaire", "7. divers"],'strip':True, 'replace':False, 'lower':True},
        'niveau_couverture_fac': {'valid_values':['base', 'option1', 'option2', 'conf', 'confort', 'plus', 'confort+'],'strip':True, 'replace':True, 'lower':True},
        'cat_assuré': {'valid_values':['actif', 'actifs', 'autre', 'chômeur', 'retraité', 'portabilité', 'loi evin', 'ayants-droits', 'suspendu'],'strip':True, 'replace':False, 'lower':True}
    }
    
    for col, config in valid_values_config.items():
        if col in df.columns:
            
            store_result(col, check_valid_values(df, col, **config, warn_type='warning'), resultats)
                
                
    ##------------------- SIREN---------------------
    if assureur == 'aesio':
        if 'siren' in df.columns:
            
            store_result('siren', check_id_length(df, 'siren', 9, skip_na=True, warn_type='warning'), resultats)
    
    ##------------------- id_assuré---------------------
    if 'id_assuré' in df.columns:
        if assureur in ['ag2r', 'macif']:
            
            store_result('id_assuré', check_id_length(df, 'id_assuré', length=8, warn_type='warning'), resultats)

        if assureur == 'aesio':
            if dft_1_raw is not None:
                
                store_result('id_assuré', check_ids_presence(dft_1, df, 'id_assuré', warn_type='warning'), resultats)
            
            store_result('id_assuré', check_special_chars(df, 'id_assuré', ['/', '\\', '_', '-'], warn_type='warning'), resultats)

            
    ##------------------- id_bénéf---------------------
    if 'id_bénéf' in df.columns:
        if assureur in ['ag2r', 'macif']:
            
            store_result('id_bénéf', check_id_length_and_content(df, 'id_bénéf', length=16, required_chars=['M', 'F'], warn_type='warning'), resultats)
        
        if assureur == 'aesio':
            if dft_1_raw is not None:
                
                store_result('id_bénéf', check_ids_presence(dft_1, df, 'id_bénéf', warn_type='warning'), resultats)
            
            store_result('id_bénéf', check_special_chars(df, 'id_bénéf', ['/', '\\', '_', '-'], warn_type='warning'), resultats)
                
    ##------------------- FR---------------------
    if ('FR' in df.columns) and ('Base_SS' in df.columns) and ('Taux_SS' in df.columns):
        
        store_result('FR', check_fr_base_taux(df, fr_col='FR', base_ss_col='Base_SS', taux_ss_col='Taux_SS', warn_type='warning'), resultats)
                
    ##------------------- Taux_SS---------------------
    if 'Taux_SS' in df.columns:
        
        store_result('Taux_SS', check_taux_ss(df, 'Taux_SS', return_col_corr=False, return_df=False, warn_type='warning'), resultats)
            
    ##------------------- R_SS---------------------
    if 'R_SS' in df.columns:
        
        store_result('R_SS', check_r_ss_equality(df, r_ss_col='R_SS', base_ss_col='Base_SS', taux_ss_col='Taux_SS', tolerance=1, warn_type='warning' ), resultats)

        store_result('R_SS', check_r_ss_without_base_taux(df, r_ss_col='R_SS', base_ss_col='Base_SS', taux_ss_col='Taux_SS', warn_type='warning'), resultats)
    
    ##------------------- RàC checks---------------------
    if 'RàC' in df.columns:
        
        store_result('RàC', check_rac_negative(df, 'RàC', warn_type='warning'), resultats)
    
    ##------------------- Quantité acte checks---------------------
    if 'quantité_acte' in df.columns:
        
        store_result('quantité_acte', check_quantite_acte(df, 'quantité_acte', warn_type='warning'), resultats)
        
        #return restructure_results(resultats)
        
    return resultats


def check_cotisations(assureur, df_raw, dates, dft_1_raw=None, rename_dict=None, raise_err=False):
    
    assureur = assureur.lower()
    if assureur not in ['aesio', 'ag2r', 'macif']:
        raise ValueError("assureur doit etre 'aesio', 'ag2r' ou 'macif'")
    
    try:
        date_debut, date_fin = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    except Exception:
        raise ValueError("dates doit etre une liste de deux dates au format YYYY-MM-DD")
    
    resultats = {}
    mandatory_cols = ['id_ent', 'date_adh', 'date_sortie', 'tr_surv', 'annee_paiement', 'mois_paiement', 'régime', 'cat_assuré', 'niveau_couverture_fac', 'structure_cot', 'cot_TTC', 'annee_comptable', 'annee_surv', "base_TTC", "option_TTC"]
    

    
    if assureur == 'aesio':
        mandatory_cols.append('siren')
    
    # Rename data
    df = rename_func(df_raw, type_fichier='santé',  type_bdd='cotisations', assureur=assureur, rename_dict=rename_dict, )
    
    if dft_1_raw is not None:
        dft_1 = rename_func(dft_1_raw, type_fichier='santé', type_bdd='cotisations', assureur=assureur, rename_dict=rename_dict)
        
    
    # Check duplicate columns
    df = remove_duplicate_columns(df)
    
    # check for missing cols
    check_mandatory_cols(df, mandatory_cols, df_name='df1', raise_err=raise_err)
    
    
    
    ## ==========================Check missing values========================
    # check for missing values
    for col in mandatory_cols:
        if col in df.columns:
            resultats[col] = []
            
            if col in ['siren', 'mois_paiement']:
                warn_type='alert'
            elif (col == 'id_ent') & (assureur != 'aesio'):
                warn_type='alert'
            else:
                warn_type='warning'
                
            store_result(col, check_missing(df, col, warn_type=warn_type), resultats)
                
                
    ## ==========================Check numbers========================
    check_cols = ['cot_TTC', "base_TTC","option_TTC"]
    for col in check_cols:
        if col in df.columns:
            
            store_result(col, chek_col_float(df, col, warn_type='alert'), resultats)
            
    
    ## ==========================Dates checks========================
    date_checks = {
        'mois_paiement':["%Y-%m", "%m", "%Y%m"], 
        'annee_surv':["%Y"], 
        'annee_paiement':["%Y"], 
        'annee_comptable':["%Y"],
        'date_sortie':None,
        'date_adh':None,
        # 'date_debut_periode':None
    }
    
    for col, format in date_checks.items():
        if col in df.columns:
            
            store_result(col, check_date_formats(df, col, date_patterns=format,  warn_type='warning'), resultats)
                
    ## ==========================Check of valid values======================== cols:[niveau_couverture_fac', 'cat_assuré']
    valid_values_config={
        'niveau_couverture_fac': {'valid_values':['base', 'option1', 'option2', 'conf', 'confort', 'plus', 'confort+'],'strip':True, 'replace':True, 'lower':True},
        'cat_assuré': {'valid_values':['actif', 'actifs', 'autre', 'chômeur', 'retraité', 'portabilité', 'loi evin', 'ayants-droits', 'suspendu'],'strip':True, 'replace':False, 'lower':True}
    }
    
    for col, config in valid_values_config.items():
        if col in df.columns:
            
            store_result(col, check_valid_values(df, col, **config, warn_type='warning'), resultats)
                
    ## --------------------------- siren----------------            
    if assureur == 'aesio':
        if 'siren' in df.columns:
            
            store_result('siren', check_id_length(df, 'siren', 9, skip_na=True, warn_type='warning'), resultats)
                
    ## --------------------------- trimestre survenance----------------
    if 'tr_surv' in df.columns:
        patterns = [
            r'^\d{4}t[1-4]$',   # "YYYYt(1-4)"
            r'^\d{4}T[1-4]$',   # "YYYYT(1-4)"
            r'^t[1-4]\d{4}$',   # "t(1-4)YYYY"
            r'^T[1-4]\d{4}$',   # "T(1-4)YYYY"
            r'^\d{4}[1-4]$',    # "YYYY(1-4)"
        ]
        
        store_result('tr_surv', check_trimester_format(df, col='tr_surv', patterns=patterns, warn_type='warning'), resultats)
                
    return resultats

@st.cache_data
def id_verif(df_prest_cot_raw, df_effectifs_raw, type_bdd, rename=True, rename_dict=None, inverse=False):
    """
    Check if the unique ids in 'prestations' are present in 'effectifs' (or vice versa if inverse=True).
    """
    
    # if rename:
    #     # Rename data
    #     df_prest_cot_raw = rename_func(df_prest_cot_raw, type_fichier='santé', type_bdd='prestations', assureur=assureur, rename_dict=rename_dict)
    #     df_effectifs_raw = rename_func(df_effectifs_raw, type_fichier='santé', type_bdd='effectifs', assureur=assureur, rename_dict=rename_dict)
    
    # Check for duplicate columns
    df_prestations = remove_duplicate_columns(df_prest_cot_raw)
    df_effectifs = remove_duplicate_columns(df_effectifs_raw)
    
    # Columns to check for unique IDs
    cols = ['id_bénéf', 'id_assuré', 'id_ent', 'siren']
    results = {col: [] for col in cols if ((col in df_prestations.columns) and (col in df_effectifs.columns))}
    
    for col in cols:
        if (col in df_prestations.columns) and (col in df_effectifs.columns):
            # Get unique IDs from both datasets for the column
            unique_prest_ids = df_prestations[col].dropna().unique()  # Unique IDs in 'prestations'
            unique_eff_ids = df_effectifs[col].dropna().unique()      # Unique IDs in 'effectifs'
            
            # Check for missing IDs
            if not inverse:
                missing_ids = set(unique_prest_ids) - set(unique_eff_ids)  # IDs in prestations but not in effectifs
                total_ids = len(unique_prest_ids)  # Total unique ids in prestations
            else:
                missing_ids = set(unique_eff_ids) - set(unique_prest_ids)  # IDs in effectifs but not in prestations
                total_ids = len(unique_eff_ids)    # Total unique ids in effectifs
            
            # If there are missing IDs, calculate the percentage and add the warning message
            if missing_ids:
                missing_count = len(missing_ids)
                percentage = (missing_count / total_ids) * 100 if total_ids > 0 else 0
                
                if not inverse:
                    message = { 'warning': f'{missing_count}  ids uniques {col} en {type_bdd.title()} ne sont pas presents en Effectifs, {percentage:.2f}% du total des id uniques en {type_bdd}.' }
                else:
                    message = { 'warning': f'{missing_count} ids uniques {col} en Effectifs ne sont pas presents en {type_bdd.title()}, {percentage:.2f}% du total des id uniques en Effectifs.' }
                
                results[col].append(message)
    
    # Clean up empty results (those with no warnings)
    results = {col: warnings if warnings else None for col, warnings in results.items()}
    
    return results

@st.cache_data
def controle(df, type_bdd, assureur, dates, dft_1_raw=None, rename_dict=None, raise_err=False):
    if type_bdd.lower() == 'effectifs':
        return check_effectifs(assureur=assureur, df_raw=df, dft_1_raw=dft_1_raw, rename_dict=rename_dict, raise_err=raise_err, results_by_type=False)
    elif type_bdd.lower() == 'cotisations':
        return check_cotisations(assureur=assureur, df_raw=df, dft_1_raw=dft_1_raw, rename_dict=rename_dict, raise_err=raise_err, dates=dates)
    elif  type_bdd.lower() == 'prestations':
        return check_prestations(assureur=assureur, df_raw=df, dft_1_raw=dft_1_raw, rename_dict=rename_dict, raise_err=raise_err, dates=dates)
    else:
        raise ValueError('Type de base de données non reconnu')

#________________________________________________________ Effectifs controle_____________________________________________________________


#________________________________________________________ APP Specific functions _____________________________________________________________

@st.cache_data
def load_file_preview(file, nrows=None, dtype=None, usecols=None):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls') or file.name.endswith('.xlsb'):
        return pd.read_excel(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, engine="calamine", usecols=usecols)
    elif file.name.endswith('.pkl') or file.name.endswith('.pickle'):
        return pd.read_pickle(file)
    else:
        raise ValueError("Unsupported file format")
    
@st.cache_data
def preview_file(df, nrows=50, height=250):
    st.dataframe(df.head(nrows), height=height)

@st.cache_data
def previw_uploaded_files(uploaded_files):
    dfs = []
    if uploaded_files:
        if isinstance(uploaded_files, list):
            for uploaded_file in uploaded_files:
                uploaded_file.seek(0)
                df = load_file_preview(uploaded_file, nrows=50)
                dfs.append(df)
                st.write(f"**{uploaded_file.name}**")
                preview_file(df, nrows=50)
        else:
                uploaded_files.seek(0)
                df = load_file_preview(uploaded_files, nrows=50)
                dfs.append(df)
                st.write(f"**{uploaded_files.name}**")
                preview_file(df, nrows=50)
    return dfs
    
def estimate_csv_rows(file):
    file.seek(0)
    for i, line in enumerate(file):
        pass
    file.seek(0)  # Reset file pointer to the beginning
    return i

def load_file(file, nrows=None, dtype=None, usecols=None):
    CHUNK_SIZE = 5000  # Number of rows per chunk

    if file.name.endswith('.csv'):
        total_rows = nrows if nrows is not None else estimate_csv_rows(file)
        
        st.write(f'Chargement du fichier {file.name}....')  # Display label
        progress_bar = st.progress(0)
        chunks = []
        rows_processed = 0

        # Reset file pointer to the beginning for reading
        file.seek(0)
        chunk_iter = pd.read_csv(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, usecols=usecols, chunksize=CHUNK_SIZE)
        
        for chunk in chunk_iter:
            chunks.append(chunk)
            rows_processed += len(chunk)
            progress_bar.progress(min(1.0, rows_processed / total_rows))
            if rows_processed >= total_rows:
                break

        progress_bar.progress(1.0)
        return pd.concat(chunks, ignore_index=True)

    elif file.name.endswith('.xlsx') or file.name.endswith('.xls') or file.name.endswith('.xlsb'):
        with st.spinner(f'Chargement du fichier {file.name}....'):
            # Streamlit file uploader provides an in-memory buffer
            return pd.read_excel(file, nrows=nrows, dtype=dtype, na_values="@", keep_default_na=True, engine="calamine", usecols=usecols)
    elif file.name.endswith('.pkl') or file.name.endswith('.pickle'):
        with st.spinner(f'Chargement du fichier {file.name}....'):
            file.seek(0)
            return pd.read_pickle(file)
    else:
        raise ValueError("Unsupported file format")
    
@st.cache_resource
def init_appearance(logo, title):
    
    #logo
    # st.logo(logo, icon_image=logo)
    
    # Separation
    st.divider()

    log, titl = st.columns([1,2])
    
    # log.image(logo, width=200)
    
    # Titre de l'application
    titl.title(title)

    # Separation
    st.divider()

# Function to calculate quarter start and end dates based on an offset
def get_quarter_dates(offset=0, date=None):
    """
    Get the start and end dates of the current, previous, or next quarter, based on the offset.
    
    Args:
    - offset (int): Offset for the quarter (-1 for previous, 1 for next, etc.).
    - date (datetime): The reference date (defaults to today).
    
    Returns:
    - (start_date, end_date): Start and end dates of the specified quarter.
    """
    
    
    if date is None:
        date = datetime.datetime.today()

    # Calculate the current quarter (1-4)
    quarter = (date.month - 1) // 3 + 1
    # Adjust quarter based on offset (-1 for previous, 1 for next)
    quarter += offset
    
    # Adjust the year if needed
    year = date.year + (quarter - 1) // 4
    quarter = (quarter - 1) % 4 + 1  # Keep quarter between 1 and 4
    
    # Calculate start and end dates of the quarter
    start_date = datetime.date(year, 3 * quarter - 2, 1)
    if quarter < 4:
        end_date = datetime.date(year, 3 * quarter + 1, 1) - datetime.timedelta(days=1)
    else:
        end_date = datetime.date(year, 12, 31)

    return start_date, end_date

def get_interactive_quarters():
    # Initialize session state for quarter offset if not already set
    if 'quarter_offset' not in st.session_state:
        st.session_state.quarter_offset = 0  # Start with the current quarter

    # Layout: Display the current quarter dates and provide buttons for navigation
    col1, col2, col3 = st.columns([1, 3, 1])

    # Previous Quarter Button
    col1.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
    with col1:
        if st.button("Trimestre précédent"):
            st.session_state.quarter_offset -= 1  # Move one quarter back

    # Show the current quarter's date range
    with col2:
        # Get the quarter dates based on the current offset
        start_date, end_date = get_quarter_dates(offset=st.session_state.quarter_offset)
        
        # Display the date input with the quarter's start and end dates
        dates = st.date_input(
            "Select the control period",
            (start_date, end_date),
            format="DD-MM-YYYY"
        )

    # Next Quarter Button
    col3.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
    with col3:
        if st.button("Trimestre suivant"):
            st.session_state.quarter_offset += 1  # Move one quarter forward

    return dates

# def get_quarter_dates(date=None):
    
#     if date is None:
#         date = datetime.today()
        
#     quarter = (date.month - 1) // 3 + 1
#     start_date = date(date.year, 3 * quarter - 2, 1)
#     if quarter < 4:
#         end_date = date(date.year, 3 * quarter + 1, 1) - pd.DateOffset(days=1)
#     else:
#         end_date = date(date.year, 12, 31)
#     return start_date, end_date



def get_dtypes(rename_dict):
    # Invert the rename_dict to map new column names to original names
    inv_rename_dict = {v: k for k, v in rename_dict.items()}
    
    # Initialize an empty dictionary to store the data types
    dtypes = {}
    
    # List of columns that should be treated as strings
    str_cols = [
        'id_ent', 'id_assuré', 'id_bénéf', 'type_bénéf', 
        'date_adh_cat', 'date_sortie_cat', 'date_adh_bénéf', 
        'date_sortie_bénéf', 'sexe', 'code_postal', 'tr_surv', 
        'mois_paiement', 'siren', 'siret'
    ]
    
    # Iterate over the list of string columns
    for col in str_cols:
        # Check if the original column name exists in the rename dictionary
        if col in inv_rename_dict:
            # Map the original column name to 'str' in the dtypes dictionary
            original_col_name = inv_rename_dict[col]
            dtypes[original_col_name] = 'str'
    
    return dtypes

def get_types_assureurs(type_fichier, json_path):
    from os import path
    from json import load
    
    if path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = load(file)
        
        # Extracting the types of BDD
        bdd_types = list(data.get(type_fichier, {}).keys())
        
        # Extracting the assureurs
        assureurs = set()
        for bdd in data.get(type_fichier, {}).values():
            assureurs.update(bdd.keys())
        
        return bdd_types, list(assureurs)
    else:
        st.error('JSON file not found')
        
# def add_assureur(json_path, bdd_type, assureur):
#     from os import path
#     from json import load, dump
    
#     if path.exists(json_path):
#         # Load the existing JSON data
#         with open(json_path, 'r', encoding='utf-8') as file:
#             data = load(file)
        
#         # Check if the bdd_type exists
#         if bdd_type not in data:
#             data[bdd_type] = {}
        
#         # Add the assureur with an empty dictionary if it doesn't exist
#         if assureur not in data[bdd_type]:
#             data[bdd_type][assureur] = {}
#         else:
#             print(f"L'assureur '{assureur}' existe déjà dans le type de BDD '{bdd_type}'.")

#         # Save the updated JSON data back to the file
#         with open(json_path, 'w', encoding='utf-8') as file:
#             dump(data, file, ensure_ascii=False, indent=4)
#     else:
#         st.error('JSON file not found')
        
def get_col_maps(type_fichier, type_bdd, assureur, json_path=r'C:\Users\Yacine AMMI\Yacine\Notebooks\AOPS\Scripts\renaming.json'):
    
    from os import path
    from json import load
    
    # Load existing rename dictionary from JSON file
    if path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            data = load(file)
            existing_dict = data.get(type_fichier, {}).get(type_bdd, {}).get(assureur, {})
            return existing_dict
    else:
        st.error('JSON file not found')
        
# Function to apply custom styles to renamed columns
def highlight_renamed(df, rename_dict_updated):
    def highlight_columns(col):
        # If the column was renamed, apply green, else red
        if col.name in rename_dict_updated.values():
            return ['background-color: lightgreen'] * len(col)
        else:
            return ['background-color: lightcoral'] * len(col)
    
    return df.style.apply(highlight_columns, axis=0)

def telechargement(result_df):
    col1, col2, col3 = st.columns([2,2,1])

    with col1:
        # Champ de texte pour le nom du fichier
        file_name = col1.text_input("Nom du fichier sans extension", "dataframe_concatene")
        
    with col2:
        # Sélection du format de téléchargement
        download_format = col2.selectbox("Choisir le format de téléchargement", ["CSV", "Excel", "Pickle"])
    
    with col3:
        col3.markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
        if st.button('Appliquer'):
            if file_name and download_format:
                with st.spinner('Chargement du fichier, \nmerci de patienter ....'):
                    buffer = BytesIO()
                    if download_format == "CSV":
                        # Téléchargement en format CSV
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Télécharger en CSV", data=csv, file_name=f'{file_name}.csv', mime='text/csv')
                    elif download_format == "Excel":
                        # Téléchargement en format Excel
                        result_df.to_excel(buffer, index=False, engine='xlsxwriter')
                        st.download_button(label="Télécharger en Excel", data=buffer, file_name=f'{file_name}.xlsx', mime='application/vnd.ms-excel')
                    elif download_format == "Pickle":
                        # Téléchargement en format Pickle
                        result_df.to_pickle(buffer)
                        st.download_button(label="Télécharger en Pickle", data=buffer, file_name=f'{file_name}.pkl', mime='application/octet-stream')
            else:
                st.error('Veuillez remplir les champs requis')

#--------------------------------------------- Mapping modalities functions ------------------------------------------------
def update_mapping(mapping_df, col):
    for index, row in mapping_df.iterrows():
        st.session_state.mappings[col][row['Ancienne']] = row['Nouvelle']

def load_mappings(file_path):
    if not path.exists(file_path):
        return {
            'niveau_couverture': {},
            'categorie_assuré': {},
            'type_bénéf': {}
        }
    
    with open(file_path, 'r', encoding='utf-8') as file:
        mappings = load(file)
    return mappings

def save_mappings(file_path, mappings):
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(mappings, file, ensure_ascii=False, indent=4)
        
def restore_editions(session_state_df, session_state_dict):
    for index, updates in session_state_dict["edited_rows"].items():
        for key, value in updates.items():
            session_state_df.loc[session_state_df.index == index, key] = value
    return pd.DataFrame(session_state_df)

def mise_en_forme_df(df, mappings, col):
    
    if col in df.columns:
        
        df_formatted = df.copy()
        mapper = mappings[col]
        
        maps_not_in_mapper = df.loc[
            df[col].notna() & 
            (~df[col].str.lower().isin(mapper.keys()))
        , col].str.lower().unique()
        
        if len(maps_not_in_mapper) > 0:
            warning(f"""{col} mapping interompu! Les valeurs {maps_not_in_mapper} ne sont pas disponibles!
            Veuillez modifier le dictionnaire des valeurs ou faire la transcodification manuellement.""")
        else:
            df_formatted[col] = df[col].str.lower().map(mapper)
            #success(f"{col} mapping avec succès!")
    
    return df_formatted

def edit_mapping(df, col, mapper):
    # Create a categorical series from the lowercase values
    cat_series = pd.Categorical(df[col].str.lower())

    # Create a mapping array
    unique_cats = cat_series.categories
    mapping_array = np.array([mapper.get(cat, np.nan) for cat in unique_cats])

    # Apply the mapping
    mapped_values = mapping_array[cat_series.codes]

    # Create the result DataFrame
    return pd.DataFrame({ 'Ancienne': cat_series, 'Nouvelle': mapped_values }).drop_duplicates()

# Function to check if all mappings are complete
def check_mappings_complete(mapping_df):
    return all(mapping_df['Nouvelle'].notna())

def handle_column_mapping(df, col_to_map, st_col, session_state, MAPPINGS_FILE):

    session_state[f"{col_to_map}_edited_df"] = edit_mapping(
        session_state.df, col_to_map, session_state.mappings[col_to_map]
    ).reset_index(drop=True)

    st_col.write(f"**{col_to_map}**")
    
    edited_df = st_col.data_editor(
        session_state[f"{col_to_map}_edited_df"],
        column_config={
            'Ancienne': st.column_config.TextColumn(disabled=True, required=True),
            'Nouvelle': st.column_config.SelectboxColumn(required=True, options=set(session_state.mappings[col_to_map].values()))
        },
        hide_index=True,
        key=f'{col_to_map}_edited_dict'
    )

    if st_col.button(f'Mettre en forme {col_to_map}'):
        edited_df = restore_editions(session_state[f"{col_to_map}_edited_df"], session_state[f"{col_to_map}_edited_dict"])
        
        if check_mappings_complete(edited_df.replace('nan', None)):
            update_mapping(edited_df, col_to_map)
            save_mappings(MAPPINGS_FILE, session_state.mappings)
            st_col.success(f"{col_to_map} mis en forme avec succès.")
            st.session_state.df = mise_en_forme_df(session_state.df, session_state.mappings, col_to_map)
            st.session_state.final_df = st.session_state.df.copy()
            # session_state.mapped_cols.append(col_to_map)
            
        else:
            st_col.error(f"Veuillez compléter tous les mappings pour {col_to_map} avant de continuer.")


mandatory_cols = {
    "santé": {
        "effectifs": {
            "id_ent": "text",
            "id_assuré": "text",
            "id_bénéf": "text",
            "siren": "text",
            "siret": "text",
            "type_bénéf": "text",
            "date_naissance": "date",
            "niveau_couverture_oblg": "text",
            "niveau_couverture_fac": "text",
            "cat_assuré": "text",
            "date_adh_cat": "date",
            "date_sortie_cat": "date",
            "date_adh_bénéf": "date",
            "date_sortie_bénéf": "date",
            "sexe": "text",
            "code_postal": "text",
            "régime": "text",
        },
        "cotisations": {
            "id_ent": "text",
            "id_assuré": "text",
            "id_bénéf": "text",
            "siren": "text",
            "siret": "text",
            "date_adh": "date",
            "date_sortie": "date",
            "tr_surv": "text",
            "annee_comptable": "integer",
            "annee_surv": "integer",
            "mois_paiement": "text",
            "régime": "text",
            "cat_assuré": "text",
            "type_bénéf": "text",
            "niveau_couverture_oblg": "text",
            "niveau_couverture_fac": "text",
            "cot_TTC": "float",
            "base_TTC": "float",
            "option_TTC": "float",
            "option_oblg_TTC": "float",
            "option_fac_TTC": "float",
        },
        "prestations": {
            "id_ent": "text",
            "id_assuré": "text",
            "id_bénéf": "text",
            "siren": "text",
            "siret": "text",
            "type_bénéf": "text",
            "niveau_couverture_oblg": "text",
            "niveau_couverture_fac": "text",
            "cat_assuré": "text",
            "regime": "text",
            "sexe": "text",
            "code_acte": "text",
            "libellé_acte": "text",
            "famille_acte": "text",
            "quantité_acte": "integer",
            "date_soins": "date",
            "date_paiement": "date",
            "FR": "float",
            "Base_SS": "float",
            "Taux_SS": "float",
            "R_SS": "float",
            "RC_Base": "float",
            "RC_Option": "float",
            "RC_Autre": "float",
            "RàC": "float",
        },
    },
    "prévoyance": {
        "prestations": {
            "id_ent": "text",
            "id_assuré": "text",
            "id_bénéf": "text",
            "siren": "text",
            "siret": "text",
            "date_naissance":"date",
            "date_surv":"date",
            "date_comptable":"date",
            "date_debut_indemn":"date",
            "date_fin_indemn":"date",
            "prest_TTC":"float"
            }, 
        "cotisations": {
            "id_ent": "text",
            "id_assuré": "text",
            "id_bénéf": "text",
            "siren": "text",
            "siret": "text",
            "annee_comptable": "integer",
            "annee_surv": "integer",
            "tr_surv": "texte",
            "mois_compatble": "texte",
            "catégorie": "texte",
            "poste": "texte",
            "code_postal": "texte",
            "cot_TTC": "float"
            }},
}


def convert_dtypes(df, type_fichier, type_bdd, raise_warning=True):
    """
    Convert DataFrame columns to their corresponding data types based on mandatory_cols.

    Args:
    - df (pd.DataFrame): The DataFrame to be processed.
    - type_bdd (str): The type of dataset (e.g., "effectifs", "cotisations", "prestations").
    - assureur (str): An optional string representing the assureur if needed.
    
    Returns:
    - pd.DataFrame: The DataFrame with columns converted to their respective types.
    """
    
    # Fetch column types for the given type_bdd
    column_types = mandatory_cols.get(type_fichier, {}).get(type_bdd, {})
    
    for col, dtype in column_types.items():
        if col in df.columns:
            if dtype == "integer":
                try:
                    df[col] = pd.to_numeric(df[col]).astype('float')
                except ValueError:
                    if raise_warning:
                        st.error(f'Error converting {col} to integer. Excluding the column from further analysis')
            elif dtype == "float":
                try:
                    df[col] = pd.to_numeric(df[col]).astype('float')
                except ValueError:
                    if raise_warning:
                        st.error(f'Error converting {col} to float. Excluding the column from further analysis')
            elif dtype == "date":
                try:
                    df[col] = correction_dates_integrale(df, col)
                except ValueError:
                    if raise_warning:
                        st.error(f'Error converting {col} to date. Excluding the column from further analysis')

            # elif dtype == "text":
            #     if df[col].dtype != 'object':
            #         df[col] = df[col].astype('str')
    
    return df

def merge_codification_aops(df, assureur, codification_file_path=r"C:\Users\Yacine AMMI\Yacine\Utilities\Codifications Actes_v170524.xlsx"):
    """
    Merge the codification of AOPs with the DataFrame.
    """
    # Load the codification file
    codification_df = pd.read_excel(
        codification_file_path,
        usecols=[
            "Assureur",
            "Code acte",
            "Famille acte AOPS v2",
            "Sous famille",
            "100% santé",
            "Sous famille 2",
            "Verres",
        ],
        engine="calamine"
    )
    codification_df['Assureur'] = codification_df['Assureur'].str.lower()
    
    df['Assureur'] = assureur
    merged_df = pd.merge(df, codification_df, how='left', left_on=['Assureur','code_acte'], right_on=['Assureur','Code acte'])
    return merged_df.rename(
        {
            "Assureur": "assureur",
            "Famille acte AOPS v2": "famille_acte_aops",
            "Sous famille": "sous_famille",
            "100% santé": "100%_santé",
            "Sous famille 2": "sous_famille_2",
            "Verres": "verres",
        },
        axis=1,
    ).drop("Code acte", axis=1)

#______________________________________________________ Rendering functions ____________________________________________________________
def display_unique_values(df, columns):
    """
    Displays unique values of selected columns in a concise format using Streamlit.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.
    columns (list): List of columns to display unique values for.
    """
    # Loop through each selected column
    for col in columns:
        if col in df.columns:
            with st.expander(f"Unique values in '{col}'"):
                unique_values = df[col].unique()
                unique_count = len(unique_values)

                # If unique values are too many, show a few, otherwise show all
                if unique_count > 10:
                    st.write(f"Showing 10 out of {unique_count} unique values:")
                    st.write(unique_values[:10])
                else:
                    st.write(f"All {unique_count} unique values:")
                    st.write(unique_values)
    
def render_values(uniques, max_display=10, display_subtitle=True):
    """
    Renders unique values as badges with a limit of how many to display.
    
    Args:
    - uniques (array-like): List of unique values to display.
    - max_display (int): Maximum number of unique values to show. Default is 10.
    """
    # from random import choices

    if len(uniques) > max_display:
        # Show a random selection of `max_display` values
        if display_subtitle:
            st.markdown(f"<p style='color: gray;'>Affichage de {max_display} sur {len(uniques)} valeurs uniques :</p>", unsafe_allow_html=True)
        selected_uniques = np.random.choice(uniques, size=max_display, replace=False)
    else:
        # Show all values if the length is <= max_display
        if display_subtitle:
            st.markdown(f"<p style='color: gray;'>Toutes les {len(uniques)} valeurs uniques :</p>", unsafe_allow_html=True)
        selected_uniques = uniques

    # Render unique values as styled badges
    unique_items = " ".join([f"<span style='background-color:#f0f0f0;border-radius:5px;padding:3px 8px;margin:2px;display:inline-block'>{val}</span>" for val in selected_uniques])
    
    # Display unique values as inline badges
    st.markdown(unique_items, unsafe_allow_html=True)

def display_uniques(df, col):
    """
    Display unique values for a given column in a DataFrame with enhanced styling.
    Also, show the column type and missing value count.
    """
    
    # Check if the column exists in the DataFrame
    if col in df.columns:
        # Get column data type
        col_dtype = df[col].dtype
        
        # Handle datetime columns directly during the unique extraction
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            uniques = df[col].dropna().dt.strftime('%Y-%m-%d').unique()
        else:
            uniques = df[col].unique()
    
        # Render header for column name
        render_header(col, font_size='24px')  # Assuming you've added font size to render_header
        
        st_cols = st.columns(2)
        
        with st_cols[0]:
            # Column type display
            st.markdown(f"<p style='color:#6c757d;'>Type de colonne: <span style='color:#17A2B8;'>{col_dtype}</span></p>", unsafe_allow_html=True)
        
        with st_cols[1]:
            render_missing_values(df, col)
        
        render_values(uniques, max_display=10, display_subtitle=True)

def display_date_summary(df, col, kind='line'):
    """Display date summary for a given column in a DataFrame with proper date formatting on x-axis."""
    
    if col in df.columns:
        # Check if the column is of datetime type, convert if necessary
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            with st.container(border=True):
            # st.markdown(f"<h4><span style='color:#173A64'>{col}</span></h4>", unsafe_allow_html=True)
                render_header(col)
            
                # Get basic statistics
                min_date = df[col].min()
                max_date = df[col].max()
                unique_dates = df[col].nunique()
                
            
                st_cols = st.columns(4)
                with st_cols[0]:
                    render_stat("Min Date", min_date.strftime('%d-%m-%Y'), value_format='')
                
                with st_cols[1]:
                    render_stat("Max Date", max_date.strftime('%d-%m-%Y'), value_format='')
                    
                with st_cols[2]:
                    render_stat("Unique Dates", unique_dates, value_format = ",.0f")
                
                with st_cols[3]:
                    render_missing_values(df, col)
            
            # Optional: Display the number of entries per month
            
                # Group by month and count occurrences
                df_grouped = df.groupby(df[col].dt.to_period('M')).size()
                df_grouped.index = df_grouped.index.to_timestamp()  # Convert PeriodIndex back to datetime
                
                if kind=='line':
                    st.line_chart(df_grouped)
                elif kind=='bar':
                    st.bar_chart(df_grouped)
                


def display_amount_histogram(df, col):
    """Display histogram or density plot for numerical columns."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        # st.markdown(f"<h4>Distribution of {col}</h4>")
        
        # Using seaborn for better visualization
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'{col}')
        plt.ylabel('Frequency')
        st.pyplot(plt)

def display_amount_boxplot(df, col):
    """Display box plot for numerical column."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        # st.markdown(f"<h4>Box Plot of {col}</h4>")
        sns.set_style(style="whitegrid")
        plt.figure(figsize=(6, 8))
        sns.boxenplot(y=df[col])
        plt.title(f'Box Plot de {col}')
        st.pyplot(plt)

                
def display_amount_summary(df, col):
    """Display summary statistics for a given numerical column in a DataFrame."""
    
    if col in df.columns:
        with st.container(border=True):
            
            # Ensure the column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Basic statistics
                sum_val = df[col].sum()
                mean_val = df[col].mean()
                median_val = df[col].median()
                min_val = df[col].min()
                max_val = df[col].max()
                # std_val = df[col].std()
                quantiles = df[col].quantile([0.25, 0.5, 0.75])
                
                st_cols = st.columns(2)
                
                with st_cols[0]:
                    render_header(col)
                    render_stat("Somme", sum_val)
                    render_stat("Moyenne", mean_val)
                    render_stat("Médiane", median_val)
                    render_stat("Min", min_val)
                    render_stat("Max", max_val)
                    render_stat("25e Percentile", quantiles[0.25])
                    render_stat("75e Percentile", quantiles[0.75])
                    
                    # render null_val with different color
                    render_missing_values(df, col)

                # st_cols = st.columns(3)
                # Optional: Display a distribution of values (histogram)
                with st_cols[1]:
                    display_amount_boxplot(df, col)
                    # st.bar_chart(df[col].value_counts(bins=10).sort_index())
                    
                # with st_cols[0]:
                #     display_amount_histogram(df, col)
            else:
                st.write("This column is not numeric.")

def display_id_column(df, col, id_length=None):
    """
    A function to check an ID column for:
    - Missing values
    - Consistent length of values
    - Number of unique values
    
    Args:
    - df: The DataFrame containing the data.
    - col: The ID column to check.
    - id_length: The expected length of the ID values. If provided, checks if all values have this length.
    """
    
    if col == 'siren':
        id_length = 9
    elif col == 'siret':
        id_length = 14
        
    with st.container(border=True):
        render_header(f"{col}")
        
        st_cols = st.columns(2)
        
        with st_cols[0]:
            # Number of unique values
            unique_values_count = df[col].nunique()
            render_stat(label="Valeurs Uniques", value=unique_values_count, value_format=",.0f")
        
        with st_cols[1]:
            # Check missing values
            render_missing_values(df, col)
        
        chars = ['/', '\\', '_', '-']
        inconsistent_id_chars = check_special_chars(df, col, chars , return_df=True, warn_type='warning')
        
        if inconsistent_id_chars is not None:
            inconsistent_id_chars = inconsistent_id_chars[col].unique()

            render_custom_text(f'{len(inconsistent_id_chars)} IDs inconsistants contenant un charactère spécial:  [ {" , ".join(chars)} ]', color='red')
            render_values(inconsistent_id_chars)
        
        if (col == 'siret') and ('siren' in df.columns):
            
            invalid_ids = check_siren_siret(df, siren_col='siren', siret_col='siret', return_df=True)
            if invalid_ids is not None:
                invalid_ids = invalid_ids[col]
                render_custom_text(f'{len(invalid_ids.unique())} IDs SIRET uniques qui ne correspondent pas au IDs SIREN dans {len(invalid_ids):,.0f} ({(len(invalid_ids)/len(df))*100:.2f}%) lignes', color='red')
        
        # Check validity siren / siret
        if col in ['siren', 'siret']:
            # checker si il exist des sirens / siret invalid
            invalid_ids = check_validity_siren_siret(df, col_name=col, id_type=col, skip_na=False, return_df=True)
            if invalid_ids is not None:
                # invalid_ids = invalid_ids[col] problem fonction returning digits instead of strings
                invalid_ids = df.loc[invalid_ids.index, col]
                render_custom_text(f'{len(invalid_ids.unique())} {col} uniques invalides dans {len(invalid_ids):,.0f} ({(len(invalid_ids)/len(df))*100:.2f}% )lignes', color='red')
                render_values(invalid_ids.astype(str).unique())
            
        id_lengths = df[col].dropna().astype(str).str.len()
        
        # No specific length provided, just give a summary of lengths
        length_counts = id_lengths.value_counts()
        # If more than one unique length, plot distribution
        if len(length_counts) > 1:
            
            # Convert the length_counts Series to a DataFrame for Streamlit's bar_chart
            length_df = pd.DataFrame({
                'Length': length_counts.index,
                'Count': length_counts.values
            }).set_index('Length')

            # Plot using Streamlit's built-in bar chart
            st.bar_chart(length_df, x_label ='Longueur des id')
        else:
            render_custom_text(f"Tous les identifiants ont la même longueur de {id_length if id_length else length_counts.index[0]}", color='#28a745')
            # st.markdown(f"<p style='color:#28a745;'>Tous les identifiants ont la même longueur de {id_length if id_length else length_counts.index[0]}</p>", unsafe_allow_html=True)
                
def check_codification_aops(df):

    global_st_cols = st.columns(2)
                                
    with global_st_cols[0]:
        
        with st.container(border=True):
            
            render_header("Verifier les codes AOPS", font_size="24px")    
            st_cols = st.columns(2)
                                
            with st_cols[0]:
                # check the number of missing codes actes
                missing_codes = df.loc[df['famille_acte_aops'].isnull(), 'code_acte'].unique()
                
                if len(missing_codes) > 0:
                    render_stat("Codes manquants", value=len(missing_codes), label_color='red', value_color='red', value_format=',.0f')
                else:
                    render_stat("Codification", value="Tous les codes sont présents", value_color='#28a745')
                    
            with st_cols[1]:
                render_missing_values(df, 'famille_acte_aops')        
                
            if len(missing_codes) > 0:
                render_values(missing_codes)
                
def render_overview(df: pd.DataFrame):
    """
    Fournit un aperçu général du dataframe, y compris des informations de base, des données manquantes et l'utilisation de la mémoire.
    """
    with st.container(border=True):
        st_cols = st.columns(4)
            
        with st_cols[0]:
            with st.container(border=False):
            
                render_header("Aperçu Général",  font_size="24px")
                
                shape = df.shape
                #compute missing cells of df
                missing_cells = df.isnull().sum().sum()
                
                render_stat("Nombre de Lignes", value=shape[0], value_format=",.0f")
                render_stat("Nombre de Colonnes", value=shape[1], value_format=",.0f")
                if missing_cells > 0:
                    render_stat("Cellules vides", value=missing_cells, value_format=",.0f", suffix=f" ({(missing_cells/df.size)*100:.2f})%", value_color='red')

                
                            
        with st_cols[1]:
            with st.container(border=False):
                render_header("Types de Colonnes",  font_size="24px")
                
                col_types = df.dtypes.value_counts()
                for dtype, count in col_types.items():
                    render_stat(f"{dtype}", value=count, value_format=",.0f")
                    
        with st_cols[2]:
            with st.container(border=False):
                render_header("Données Manquantes",  font_size="24px")
                
                missing_data = df.isnull().mean() * 100
                missing_columns = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if not missing_columns.empty:
                    # render_custom_text(f"{len(missing_columns)} colonnes ont des valeurs manquantes :")
                    # for col, pct_missing in missing_columns.items():
                    #     # st.markdown(f"- **{col} :** {pct_missing:.2f}% de valeurs manquantes")
                    #     render_stat(f"{col}", value=pct_missing, suffix='%', value_color='red')
                    missing=missing_columns.index
                    render_stat("Colonnes a valeurs manquantes", value=len(missing), value_format=",.0f", value_color='red')
                    render_values(missing, max_display=30, display_subtitle=False)
                    
                    
                else:
                    render_custom_text("Aucune valeur manquante trouvée.", color='green')
                    # st.markdown("Aucune valeur manquante trouvée.")
    
                
        with st_cols[3]:
                with st.container(border=False):
                    render_header("Utilisation de la mémoire",  font_size="24px")
                    
                    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convertir en MB
                    render_stat("Poids de la table", value=memory_usage, suffix=' MB')
               
def compare_date_cols(df):
    """
    This function compares pairs of date columns in a DataFrame to check for inconsistencies 
    where the second date is earlier than the first. It also highlights date columns in the displayed DataFrame.
    
    Args:
    - df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    - None: Outputs directly to Streamlit.
    """
    # Define date pairs to check
    date_pairs = [
        ('date_soins', 'date_paiement'),
        ('date_adh_bénéf', 'date_adh'),
        ('date_sortie_cat', 'date_sortie_bénéf'),
        ('date_adh', 'date_sortie'),
        ('date_adh_cat', 'date_sortie_cat'),
        ('date_adh_bénéf', 'date_sortie_bénéf')
    ]
    
    # Find the first matching date pair in the DataFrame
    dates = []
    for date1, date2 in date_pairs:
        if date1 in df.columns and date2 in df.columns:
            dates.append((date1, date2))
    
    # If a pair is found, proceed with the comparison
    if len(dates) > 0:
        with st.container(border=False):
            render_header("Comparaison des dates", font_size="24px")
            
            for d in dates:
                # Perform the date comparison
                dates_df = compare_dates(df, d[0], d[1], condition='<=', return_df=True)
                
                inconsistent_dates = len(dates_df)
                if inconsistent_dates > 0:
                    render_custom_text(f"Incohérences trouvées : {inconsistent_dates} lignes où {d[1]} < {d[0]}", color='red')
                    render_custom_text("Exemple d'incohérences :")
                    
                    # Highlight the date columns in the DataFrame
                    styled_df = dates_df.style.applymap(lambda _: 'background-color: lightcoral', subset=[d[0], d[1]])
                    st.dataframe(styled_df)
                else:
                    render_custom_text(f"Pas d'incohérence trouvé : toutes les lignes respectes la condition {d[1]} >= {d[0]}", color='green')

            
    
def pivot_cot(df):
    
    pivot_df = df.pivot_table(index='mois_paiement', columns='annee_surv', values='cot_TTC', aggfunc='sum')
    with st.container(border=True):
        render_header( "Pivot des coûts", )
        render_custom_text('Pivot des coûts totaux (TTC) par mois de paiement et annee de survenance',)
        st.dataframe(pivot_df, use_container_width=True)
        
        
                    
                
def orange_markdown_string(string):
    return f"**:Black[{string}]**"

def render_stat(label: str, value: float, value_format: str = ",.2f", suffix: str = "", label_color: str = "#6c757d", value_color: str = "#17A2B8", font_size: str = "16px"):
    """
    A reusable function to render a labeled statistic with the given value and format, including customization.

    Args:
    - label (str): The label to display (e.g., 'Mean', 'Max', etc.).
    - value (float): The value to display.
    - value_format (str): How the value should be formatted. Default is 'comma, 2 decimals'.
    - suffix (str): The suffix to add after the value (e.g., '%'). Default is an empty string.
    - label_color (str): The color for the label text. Default is gray.
    - value_color (str): The color to display the value in. Default is light blue (#17A2B8).
    - font_size (str): Size of the font for the label and value. Default is '16px'.
    """
    formatted_value = f"{value:{value_format}}{suffix}"
    st.markdown(
        f"<p style='font-size:{font_size}; color:{label_color}'>{label}: <span style='color:{value_color}'>{formatted_value}</span></p>",
        unsafe_allow_html=True
    )



def render_missing_values(df, col, missing_color: str = "red", non_missing_color: str = "#28a745"):
    """
    A reusable function to render the missing values count for a given column.
    
    Args:
    - df: The DataFrame containing the data.
    - col: The column to check for missing values.
    - missing_color (str): The color to use for missing values. Default is red.
    - non_missing_color (str): The color to use when there are no missing values. Default is green.
    """
    missing_count = df[col].isnull().sum()
    ## calc missing mercentage
    missing_percentage = (missing_count / len(df)) * 100
    
    if missing_count > 0:
        st.markdown(f"<p style='color: {missing_color};'>Valeurs manquantes : {missing_count:,.0f} ({missing_percentage:.1f}%)</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: {non_missing_color};'>Aucune valeur manquante</p>", unsafe_allow_html=True)

def render_header(title: str, color: str = "#173A64", font_size: str = "28px"):
    """
    A reusable function to render a section header with a custom color and font size.
    
    Args:
    - title (str): The title of the section or column.
    - color (str): The color of the title text. Default is a dark blue (#173A64).
    - font_size (str): The size of the font. Default is '24px'.
    """
    st.markdown(f"<h4><span style='color:{color}; font-size:{font_size}'>{title}</span></h4>", unsafe_allow_html=True)
    
def render_custom_text(content: str, color: str = "#6c757d", font_size: str = "16px", bold: bool = False, italic: bool = False):
    """
    Renders custom text with given styling in Streamlit.
    
    Args:
    - content (str): The text content to display.
    - color (str): Text color. Default is 'black'.
    - font_size (str): Font size of the text. Default is '16px'.
    - bold (bool): Whether to render the text in bold. Default is False.
    - italic (bool): Whether to render the text in italic. Default is False.
    """
    # Apply bold and italic styles
    font_weight = "bold" if bold else "normal"
    font_style = "italic" if italic else "normal"
    
    # Render the text in Streamlit with HTML styling
    st.markdown(
        f"<p style='color:{color}; font-size:{font_size}; font-weight:{font_weight}; font-style:{font_style};'>{content}</p>",
        unsafe_allow_html=True
    )

            
@st.cache_data
def resume_bdd(df, type_bdd):
    
    # extract unique values from cols according to bdd_type using get() method
    # uniq_cols = cols.get(type_bdd, {}).get("uniq_cols", {})
    
    
    # date_cols = cols.get(type_bdd, {}).get("date_cols", {})
    date_cols = ["date_adh_cat", "date_sortie_cat", "date_adh_bénéf", "date_sortie_bénéf", "date_adh", "date_sortie", "date_soins", "date_paiement", "date_naissance", "date_surv", "date_comptable", "date_debut_indemn", "date_fin_indemn",]
    
    # amount_cols = cols.get(type_bdd, {}).get("amount_cols", {})
    amount_cols = [ "FR", "Base_SS", "Taux_SS", "R_SS", "RC_Base", "RC_Option", "RC_Autre", "RàC", "cot_TTC", "base_TTC", "option_TTC", "option_oblg_TTC", "option_fac_TTC", "prest_TTC"]
    
    uniq_cols = [col for col in df.columns if col not in amount_cols]  
    
    # id_cols = cols.get(type_bdd, {}).get("id_cols", {})
    id_cols = [ "siren", "siret", "id_ent", "id_assuré", "id_bénéf" ]
    
    render_overview(df)
    
    # display_uniques_toggle =  st.expander('**:orange[Afficher les valeurs uniques]**')
    display_uniques_toggle =  st.expander(orange_markdown_string('Afficher les valeurs uniques'))
    with display_uniques_toggle:
        i=0
        st_cols = st.columns(3)
        
        for col in uniq_cols:
            if col in df.columns:
                with  st_cols[i%3]:
                    with st.container(border=True):
                        display_uniques(df, col)
                i+=1
        # display_unique_values(df, uniq_cols)

    # Display date summary
    date_summary_toggle = st.expander(orange_markdown_string('Afficher les résumés des dates'))
    with date_summary_toggle:
        compare_date_cols(df)
        i = 0
        st_cols = st.columns(2)
        for col in date_cols:
            if col in df.columns:
                with st_cols[i % 2]:
                    display_date_summary(df, col)
                    i += 1
    
    if type_bdd != 'effectifs':
        # Display cot summary
        mtn_summary_toggle = st.expander(orange_markdown_string(f'Afficher les résumés des {type_bdd}'))
        with mtn_summary_toggle:
            st_cols = st.columns(2)
            i=0
            if ('cot_TTC' in df.columns) and ('mois_paiement' in df.columns) and ('annee_surv' in df.columns):
                with  st_cols[i%2]:
                    pivot_cot(df)
                    i+=1

            for col in amount_cols:
                if col in df.columns:
                    with  st_cols[i%2]:
                        display_amount_summary(df, col)
                    i+=1

    # Display id summary
    id_summary_toggle = st.expander(orange_markdown_string('Afficher les résumés des identifiants'))
    with id_summary_toggle:
        i=0
        st_cols = st.columns(2)
        for col in id_cols:
            if col in df.columns:
                with  st_cols[i%2]:
                    display_id_column(df, col)
                i+=1
    
    # Check codification AOPS
    if 'famille_acte_aops' in df.columns:
        codification_toggle = st.expander(orange_markdown_string('Afficher les résumés de la codification AOPS'))
        with codification_toggle:
            check_codification_aops(df)
            
            
###################################################################################
# Function to get the rename mapping from the existing column names and the new names
def get_column_rename_mapping(df, rename_dict, mandatory_cols):
    """
    Fetch and map original and new column names based on the file type, database type, and insurer.

    Args:
    - df (pd.DataFrame): DataFrame with the preview of the data.
    - type_fichier (str): The type of the file being processed.
    - type_bdd (str): The database type.
    - assureur (str): Insurer information.
    - json_path (str): Path to the JSON file with column mappings.
    - mandatory_cols (dict): Mandatory columns for the given type_fichier and type.

    Returns:
    - rename_df (pd.DataFrame): DataFrame containing the mapping of new and original column names.
    - rename_dict_updated (dict): Updated rename dictionary.
    """
    original_names = df.columns
    # new_names = get_col_maps(type_fichier=type_fichier, type_bdd=type_bdd, assureur=assureur, json_path=json_path)

    
    # Filter new names that exist in the original columns
    new_names = {k: v for k, v in rename_dict.items() if k in original_names}

    # Create a rename DataFrame for mandatory columns
    rename_df = pd.DataFrame({col: None for col in mandatory_cols}.items(), columns=['Colonne', 'Original'])
    
    # Map the 'Original' column with the new_names from 'Colonne'
    if new_names:
        rename_df['Original'] = rename_df['Colonne'].map({v: k for k, v in new_names.items()})

    return rename_df

# Function to handle the renaming using Streamlit's data editor
def column_rename_interaction(rename_df, original_names):
    """
    Interact with the user to allow renaming columns through a data editor interface.

    Args:
    - rename_df (pd.DataFrame): DataFrame with columns for renaming.
    - original_names (list): List of original column names from the DataFrame.

    Returns:
    - rename_dict_updated (dict): Updated rename dictionary from user input.
    """
    new_column_names = st.data_editor(
        rename_df,
        column_config={
            "Colonne": st.column_config.Column(label="Colonnes"),
            "Original": st.column_config.SelectboxColumn(
                label='Nom original du fichier', 
                required=False, 
                options=list(set(original_names))
            )
        },
        hide_index=True,
        # key="column_name_editor",
        use_container_width=True
    )

    # Update DataFrame with the new column names from data editor
    rename_dict_updated = dict(zip(new_column_names["Original"], new_column_names["Colonne"]))
    
    return rename_dict_updated

# Function to apply the renaming and display the preview of the data
def display_renamed_preview(df_preview, rename_dict_updated):
    """
    Apply renaming based on the user's input and display a styled preview of the renamed DataFrame.

    Args:
    - df_preview (pd.DataFrame): Original DataFrame to be previewed and renamed.
    - rename_dict_updated (dict): Dictionary containing mappings of original to new column names.
    - fn: Custom functions for column highlighting.

    Returns:
    - styled_df (pd.DataFrame): Preview of the renamed DataFrame with styling applied.
    """
    df_preview_renamed = df_preview.rename(columns=rename_dict_updated)

    # Reorder columns: renamed columns first, followed by remaining ones
    renamed_columns = [renamed for original, renamed in rename_dict_updated.items() if original not in [None, 'null', np.nan, "NaN", ""]]
    remaining_columns = [col for col in df_preview_renamed.columns if col not in renamed_columns]
    ordered_columns = renamed_columns + remaining_columns

    # Apply the styling function to highlight renamed columns
    styled_df = highlight_renamed(df_preview_renamed[ordered_columns].head(50), rename_dict_updated)

    # Display the styled DataFrame with reordered columns
    st.dataframe(styled_df, hide_index=True)
    
    return renamed_columns


# Main function that encapsulates the renaming and preview process
def process_column_renaming(df_preview, rename_dict, mandatory_cols):
    """
    Full process of renaming columns in the DataFrame and previewing the renamed DataFrame.

    Args:
    - df_preview (pd.DataFrame): DataFrame preview to be processed.
    - type_fichier (str): File type for renaming configuration.
    - type_bdd (str): Database type.
    - assureur (str): Insurer information.
    - json_path (str): Path to JSON configuration.
    - mandatory_cols (dict): Dictionary of mandatory columns.
    - fn: Custom functions (highlight, mapping retrieval, etc.)
    """
    # Fetch the rename mapping
    rename_df = get_column_rename_mapping(
        df=df_preview, 
        rename_dict=rename_dict, 
        mandatory_cols=mandatory_cols
    )

    
    # # Data editor column configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display column rename interface
        st.subheader("Renommer les colonnes")
        rename_dict_updated = column_rename_interaction(rename_df, df_preview.columns)

    with col2:
        # Display the renamed DataFrame preview
        st.subheader("Aperçu des données")
        renamed_columns = display_renamed_preview(df_preview, rename_dict_updated)
    
    return rename_dict_updated, renamed_columns

@st.cache_data
def display_warnings(warnings, title, header):
    
    with st.expander(orange_markdown_string(title)):
        render_header(header)
        
        # Create three columns for the warnings output
        col1, col2, col3 = st.columns(3)
        
        for i, (col, warn) in enumerate(warnings.items()):
            with [col1, col2, col3][i % 3]:  # Cycle through the columns
                st.write(f"**{col} :**")
                if warn:
                    for w in warn:
                        key = list(w.keys())[0]
                        value = list(w.values())[0]
                        if key == 'warning':
                            st.warning(value)
                        elif key == 'alert':
                            st.error(value)
                else:
                    st.success('Aucun problème détecté')

def upload_and_rename(title, mandatory_cols, rename_dict, json_path, types=['csv', 'xlsx', 'xls', 'xlsb', 'pkl', 'pickle'], key="uploaded_file"):
    
    charger = st.expander(orange_markdown_string(title), expanded=True)
    uploaded_file = None
    
    with charger:
        uploaded_file = st.file_uploader("", accept_multiple_files=False, type=types, key=key)
        
        if uploaded_file:
            df = None
            
            df_preview = load_file_preview(uploaded_file, nrows=50)
            
            rename_dict_updated, renamed_columns = process_column_renaming(df_preview, rename_dict, mandatory_cols)

            import_dtypes = get_dtypes(rename_dict_updated)
                
            if st.button('Valider', key=key + '_btn'):
                df = load_file(uploaded_file, dtype=import_dtypes)
                df = rename_func(
                    df,
                    type_fichier=st.session_state.type_fichier,
                    type_bdd=st.session_state.type,
                    assureur=st.session_state.assr,
                    rename_dict=rename_dict_updated,
                    keep_default_dict=False,
                    warn=False,
                    update_json=False,
                    json_file=json_path,
                )[renamed_columns]
                st.session_state['tmp'] = df  # Store in session state instead of returning
                st.session_state[key + '_btn_clicked'] = True

            if key + '_btn_clicked' in st.session_state:
                render_custom_text("Fichier importé avec succès", color="#339900")
                return st.session_state['tmp']  # Return the DataFrame from session state
            else:
                render_custom_text("Veuillez valider votre selection", color="#ffbc11")
            