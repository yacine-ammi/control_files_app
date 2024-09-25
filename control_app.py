import streamlit as st
import pandas as pd
from io import BytesIO
import functions as fn
import os


# streamlit run "C:\Users\Yacine AMMI\Yacine\Notebooks\AOPS\Scripts\Controle Fichiers\control_app.py"--server.maxUploadSize 3000


def init_session_state_data():
    st.session_state.renamed_preview = []
    st.session_state.formatted_preview = []
    st.session_state.df = None
    st.session_state.final_df = None
    st.session_state.df_1 = None
    st.session_state.warnings_bad = None

def main():
    # Get the current directory (where your app is running)
    current_dir = os.path.dirname(__file__)

    # Use relative paths to the JSON, images, and other resources
    json_path = os.path.join(current_dir, 'resources', 'renaming.json')
    MAPPINGS_FILE = os.path.join(current_dir, 'resources', 'mappings.json')
    page_ico = os.path.join(current_dir, 'resources', 'merge.png')
    logo = os.path.join(current_dir, 'resources', 'Logo_AOPS_conseil.png')
    title = 'Outil de :orange[Contrôle] des Fichiers '
    
    st.set_page_config(layout="wide", page_title='Controle des Fichiers', page_icon=page_ico)
    
    fn.init_appearance(logo, title)
    
    if 'type' not in st.session_state :
        st.session_state.type_fichier = None
        st.session_state.type = None
        st.session_state.type_risque = None
        st.session_state.assr = None
        init_session_state_data()
    
    cols = st.columns([1,2,2])
    

    # selectionner le type de fichier ["Santé", "Prevoyance"] 
    with cols[0]:
        
        st.session_state.type_fichier = st.radio('Type de fichier', ["santé", "prévoyance"],
                                             help='Sélectionner le type de fichier',
                                             horizontal=True, 
                                             index=None)
        types_dispo, assr_dispo = fn.get_types_assureurs(type_fichier=st.session_state.type_fichier, json_path=json_path)
        
    if st.session_state.type_fichier == "santé":
        
        # selectionner le type de fichier ["prestations", "cotisations", "effectifs"] 
        st.session_state.type = cols[1].selectbox(label="Selectionnez le type de fichier", options=types_dispo, placeholder=f'Type de fichier', index=None, key=f"type_w")
        if st.session_state.type is not None:
            
            # selectionner l'assureur
            st.session_state.assr = cols[2].selectbox(label="Selectionnez l'assureur", options=assr_dispo, placeholder=f'Assureur', index=None, key=f"assr_w")
            
            # with cols[2]:
            #     cols[2].markdown("<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True)
            #     # add button to add new 'assureur'
            #     with st.popover("Ajouter un assureur", help="""**A utiliser avec prudence:** \n l'asureur va etre ajouté a la base de données"""):
            #         new_assr = st.text_input("Ecrire le nom de l'assureur", placeholder="Nom de l'assureur",)
            #         if  st.button("Ajouter l'assureur"):
            #             if new_assr.lower() not in fn.get_types_assureurs(json_path):
            #                 fn.add_assureur(json_path, st.session_state.type, new_assr.lower())
            #                 # message de succès
            #                 st.success(
            #                     f"**{new_assr.lower()}** a été ajouté avec succès"
            #                 )
            #             else:
            #                 st.error(f"**{new_assr.lower()}** existe deja")

    elif st.session_state.type_fichier == "prévoyance":
        
        with cols[1]:
            st_cols = st.columns(2)
            
            # selectionner le type de fichier ["prestations", "cotisations"] 
            st.session_state.type = st_cols[0].radio(label="Selectionnez le type de fichier", options=types_dispo, index=None, horizontal=True)
            if st.session_state.type == 'prestations':
                
                # selectionner le type de risque ["prestations", "cotisations"]
                st.session_state.type_risque = st_cols[1].selectbox(label="Selectionnez le type de risque", options=['mensualisation',
                                                                                                                'incapacité',
                                                                                                                'invalidité',
                                                                                                                'décès'], 
                                                            placeholder=f'Type de risque', index=None, key="type_r")
            
            if (st.session_state.type is not None) and ((st.session_state.type_risque is not None) or (st.session_state.type == 'cotisations')):
                st.session_state.assr = cols[2].selectbox(label="Selectionnez l'assureur", options=assr_dispo, index=None, key=f"assr_w")            

    uploaded_file = None

    if st.session_state.type_fichier and st.session_state.type and st.session_state.assr:
        # Chargement des fichiers
        uploaded_file = st.file_uploader("Choisir un fichier", accept_multiple_files=False, type=['csv', 'xlsx', 'xls', 'xlsb', 'pkl', 'pickle'], 
                                        #   on_change=change_init_state
                                        )

        if uploaded_file:
            
            #----------------------------------------------  Rename the DF-----------------------------------------------------
            
            uploaded_file.seek(0)
            df_preview = fn.load_file_preview(uploaded_file, nrows=50)
            
            # # Initial rename configuration based on selected type and insurer
            # original_names = df_preview.columns
            # new_names = fn.get_col_maps(type_fichier=st.session_state.type_fichier ,type_bdd=st.session_state.type,  assureur=st.session_state.assr, json_path=json_path)
            
            # # Select only columns corresponding to our original col names
            # new_names = {k:v for k,v in new_names.items() if k in original_names}
            
            
            # rename_df = pd.DataFrame({col:None for col in fn.mandatory_cols.get(st.session_state.type_fichier, {})[st.session_state.type]}.items(), columns=['Colonne', 'Original'])
            # if new_names:
            #     # map the 'Original' col with the new_names from 'Colonne'
            #     rename_df['Original'] = rename_df['Colonne'].map({v:k for k,v in new_names.items()})
            
            # # st.write(rename_df.to_dict(orient='list'))
            
            # # rename_df['Selection'] = True
            # rename_df = rename_df[[
            #     # "Selection", 
            #     "Colonne", 
            #     "Original"
            #     ]]
            
            # # Data editor column configuration
            # col1, col2 = st.columns([1, 2])
            
            # with col1:
            #     st.subheader("Renommer les colonnes")
            #     new_column_names = st.data_editor(
            #         rename_df,
            #         # num_rows="dynamic",
            #         column_config={
            #             # 'Selection': st.column_config.CheckboxColumn(label='Select',
            #             #                                              help='Selectionner les colonnes a inclure dans le controle',
            #             #                                              width=3),
            #             "Colonne": st.column_config.Column(label="Colonnes"),
            #             "Original": st.column_config.SelectboxColumn(label='Nom original du fichier', required=False, options=(list(set(original_names))))
            #         },
            #         hide_index=True,
            #         key="column_name_editor",
            #         use_container_width=True
            #     )
            
            #     # Update DataFrame with the new column names from data editor
            #     rename_dict_updated = dict(zip(new_column_names["Original"], new_column_names["Colonne"]))
                        
            
            # with col2:
            #     st.subheader("Aperçu des données")
                
            #     # Rename the preview DataFrame based on the updated rename dictionary
            #     df_preview_renamed = df_preview.rename(columns=rename_dict_updated)

            #     # Reorder columns: renamed columns first, followed by remaining ones
            #     renamed_columns =  [renamed for original, renamed in rename_dict_updated.items() if original not in [None, 'null', nan, "NaN", ""]] #list(rename_dict_updated.values())
            #     remaining_columns = [col for col in df_preview_renamed.columns if col not in renamed_columns]
            #     ordered_columns = renamed_columns + remaining_columns
                
            #     # Apply the styling function to highlight renamed columns
            #     styled_df = fn.highlight_renamed(df_preview_renamed[ordered_columns].head(50), rename_dict_updated)

            #     # Display the styled DataFrame with reordered columns
            #     st.dataframe(styled_df, hide_index=True)
            
            rename_dict = fn.get_col_maps(type_fichier=st.session_state.type_fichier ,type_bdd=st.session_state.type,  assureur=st.session_state.assr, json_path=json_path)
            mandatory_cols = fn.mandatory_cols.get(st.session_state.type_fichier, {})[st.session_state.type]
            
            rename_dict_updated, renamed_columns = fn.process_column_renaming(df_preview, rename_dict, mandatory_cols)
            
            #------------------------------------------------- Validate and test conversions -----------------------------------------------------------------------------
            
            if st.button("Valider et importer la base complète"):
                st.session_state.renamed_preview = fn.rename_func(
                    df_preview, 
                    type_fichier=st.session_state.type_fichier,
                    type_bdd=st.session_state.type,
                    assureur=st.session_state.assr,
                    rename_dict=rename_dict_updated,
                    keep_default_dict=False,
                    warn=False,
                    update_json=True,
                    json_file=json_path,
                )
                st.session_state.renamed_preview = fn.convert_dtypes(st.session_state.renamed_preview[renamed_columns], type_fichier=st.session_state.type_fichier ,type_bdd=st.session_state.type, raise_warning=True )
                st.write(st.session_state.renamed_preview)
                
        #------------------------------------------------- Import, rename and convert the whole dataset------------------------------------------------------------------------------        
        # if st.button("Valider et importer la base compléte"):
                st.subheader('Importation de la BDD complète')
                
                import_dtypes = fn.get_dtypes(rename_dict_updated)
                
                df = fn.load_file(uploaded_file, dtype=import_dtypes)
                df = fn.rename_func(
                    df,
                    type_fichier=st.session_state.type_fichier,
                    type_bdd=st.session_state.type,
                    assureur=st.session_state.assr,
                    rename_dict=rename_dict_updated,
                    keep_default_dict=False,
                    warn=False,
                    update_json=True,
                    json_file=json_path,
                )
                
                st.session_state.df = fn.convert_dtypes(df[renamed_columns], type_fichier=st.session_state.type_fichier ,type_bdd=st.session_state.type, raise_warning=False)
            
                if (st.session_state.type_fichier == 'santé') and(st.session_state.type == 'prestations'):
                    if "code_acte" in st.session_state.df.columns:
                        try:
                            st.session_state.df = fn.merge_codification_aops(st.session_state.df, assureur=st.session_state.assr, codification_file_path=r"C:\Users\Yacine AMMI\Yacine\Utilities\Codifications Actes_v170524.xlsx")
                            st.success("Codification AOPS ajouté avec succès")

                        except Exception:
                            st.error("Erreur lors de la codification des actes AOPS")
                            
            #------------------------------------------------- Format labels ------------------------------------------------------------------------------     
            
            if st.session_state.df is not None:
                
                st.divider()
                
                format_toggle = st.toggle('Mettre en forme')
                
                if format_toggle:
                    cols_to_map = ['niveau_couverture_fac', 'niveau_couverture_oblg', 'cat_assuré', 'type_bénéf']
                    cols_available = [col for col in cols_to_map if col in st.session_state.renamed_preview]
                    
                    if any(cols_available):
                        st.header("Mise en forme")
                        if 'niv_couv_edited_df' not in st.session_state:
                            # st.session_state.formatted_preview = st.session_state.renamed_preview.copy()
                            st.session_state.niv_couv_edited_df = None
                            st.session_state.niv_couv_oblg_edited_df = None
                            st.session_state.cat_assr_edited_df = None
                            st.session_state.type_bénéf_edited_df = None
                            st.session_state.mapped_cols = []
                        
                        mappings = fn.load_mappings(MAPPINGS_FILE)

                        # if 'mappings' not in st.session_state:
                        st.session_state.mappings = mappings

                        cols = st.columns(2)
                        
                        for idx, col_to_map in enumerate(cols_available):
                            fn.handle_column_mapping(df=st.session_state.df, col_to_map=col_to_map, st_col=cols[idx % 2], session_state=st.session_state, MAPPINGS_FILE=MAPPINGS_FILE)
                        
                else:
                    st.session_state.final_df = st.session_state.df.copy()
                    
            #------------------------------------------------- Download option------------------------------------------------------------------------------
                    if st.session_state.final_df is not None:
                        st.subheader('Résultat')
                        fn.preview_file(st.session_state.final_df)
                    telechargement_expand = st.expander(fn.orange_markdown_string("Téléchargement"))
                    with telechargement_expand:
                        fn.telechargement(st.session_state.final_df)
                    
                
                
            #------------------------------------------------- General Control section------------------------------------------------------------------------------ 
            st.divider()
            if st.session_state.final_df is not None:
                st.header('Contrôle du fichier')
                with st.spinner('Chargement en cours ....'):
                    fn.resume_bdd(st.session_state.final_df, st.session_state.type)
                            
            #------------------------------------------------- BAD Control section------------------------------------------------------------------------------
                st.divider()
                control_bad = st.toggle('Contrôle spécifique a la BAD')
                if control_bad:
                    
                    st.header("Points de Contrôles BAD")
                    
            #------------------------------------------------- Upload T-1 ------------------------------------------------------------------------------                    
                    
                    mandatory_cols = ['id_bénéf', 'id_assuré', 'id_ent', 'siren']
                    rename_dict = fn.get_col_maps(type_fichier=st.session_state.type_fichier ,type_bdd=st.session_state.type,  assureur=st.session_state.assr, json_path=json_path)
                    st.session_state.df_1 = fn.upload_and_rename(title="Charger T-1", mandatory_cols=mandatory_cols, rename_dict=rename_dict, json_path=json_path, types=['csv', 'xlsx', 'xls', 'xlsb', 'pkl', 'pickle'], key='df_1_up')
                    
                    if st.session_state.df_1 is not None:
                        st.success('T-1 importé avec succès')
            
            #------------------------------------------------- Control coherence and display warnings------------------------------------------------------------------------------
                if (st.session_state.type_fichier == 'santé') and (st.session_state.type == 'prestations'):
                    st.session_state.df_eff = fn.upload_and_rename(title="Charger effectifs", mandatory_cols=mandatory_cols, rename_dict=rename_dict, json_path=json_path, types=['csv', 'xlsx', 'xls', 'xlsb', 'pkl', 'pickle'], key='df_eff_up')
                    
                    if 'df_eff' in st.session_state:
                        if st.session_state.df_eff is not None:
                            st.session_state.warnings_bad_ids = fn.eff_prest_id_verif(df_prestations_raw=st.session_state.final_df, df_effectifs_raw=st.session_state.df_eff, assureur=st.session_state.assr, rename=False, inverse=False)
                            st.session_state.warnings_bad_ids_inverse = fn.eff_prest_id_verif(df_prestations_raw=st.session_state.final_df, df_effectifs_raw=st.session_state.df_eff, assureur=st.session_state.assr, rename=False, inverse=True)

                elif (st.session_state.type_fichier == 'santé') and (st.session_state.type == 'effectifs'):
                    st.session_state.df_prest = fn.upload_and_rename(title="Charger prestations", mandatory_cols=mandatory_cols, rename_dict=rename_dict, json_path=json_path, types=['csv', 'xlsx', 'xls', 'xlsb', 'pkl', 'pickle'], key='df_prest_up')
            
            #------------------------------------------------- Control BAD and display warnings------------------------------------------------------------------------------  
                st.session_state.warnings_bad = fn.controle_bad(df=st.session_state.final_df, df_1=st.session_state.df_1)
                if  st.session_state.warnings_bad is not None:
                    fn.display_warnings(st.session_state.warnings_bad, title="Contrôles BAD", header="Résultat")        
                    if  'warnings_bad_ids' in  st.session_state :
                        fn.display_warnings(st.session_state.warnings_bad_ids, title="Prestations vs Effectifs", header="Prestations manquantes dans Effectifs ")
                    if  'warnings_bad_ids_inverse' in  st.session_state :
                        fn.display_warnings(st.session_state.warnings_bad_ids_inverse, title="Effectifs vs Prestations", header="Effectifs manquants dans Prestations" )

if __name__ == "__main__":
    main()
