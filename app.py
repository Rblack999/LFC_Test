import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
import re

DATA_PATH = Path('LFC AM Development_Cell Eng & Prod.xlsx')

RENAME_MAP = {
    'Source': 'source',
    'Date': 'date',
    'Slot Die (Microns)': 'slot_die_microns',
    'Composition % (AM/C/B)': 'composition_am_c_b',
    'Cell Type': 'cell_type',
    'ELECTRODE OR CYCLING PROTOCOL NOTE': 'note',
    'Electrode Name': 'electrode_name',
    'AM': 'am_fraction',
    'Single Sided Weight (g)': 'single_sided_weight_g',
    'Double Sided Weight (g)': 'double_sided_weight_g',
    'Cathode Weight (g)': 'cathode_weight_g',
    'Active Material (g)': 'active_material_g',
    'Single Sided Average Areal Loading (EMD mg/cm2)': 'areal_loading_mg_cm2',
    'Current Density (mA/g)': 'current_density_mA_g',
    'Current (A)': 'current_A',
    'Initial electrode thickness for average': 'initial_thickness_1',
    '': 'unused',
    'Initial Thickness (Microns)': 'initial_thickness_microns',
    'Initial Thickness (cm)': 'initial_thickness_cm',
    'Final Thickness (Microns)': 'final_thickness_microns',
    'Final Thickness (No CC)': 'final_thickness_no_cc',
    'Final Thickness (cm)': 'final_thickness_cm',
    'Volume Before (cm3)': 'volume_before_cm3',
    'Volume After (cm3)': 'volume_after_cm3',
    'Density Before (g/cm3)': 'density_before_g_cm3',
    'Density After (g/cm3)': 'density_after_g_cm3',
    'Electrode Resistance -4 Pt Probe (Ohm)': 'resistance_4pt_1',
    'Average Electrode Resistance -4 Pt Probe (Ohm)': 'resistance_4pt_avg',
    'Through Plain Resistance Measurement': 'through_plane_res_ohm',
    'Channel': 'channel',
    'Initial OCV (V)': 'initial_ocv_v',
    'Date Cycling': 'date_cycling',
}

NUMERIC_COLS = [
    'double_sided_weight_g',
    'cathode_weight_g',
    'active_material_g',
    'areal_loading_mg_cm2',
    'final_thickness_microns',
    'density_before_g_cm3',
    'density_after_g_cm3',
    'resistance_4pt_avg',
    'through_plane_res_ohm',
]

METRIC_LABELS = {
    'double_sided_weight_g': 'Double Sided Weight (g)',
    'cathode_weight_g': 'Cathode Weight (g)',
    'active_material_g': 'Active Material (g)',
    'areal_loading_mg_cm2': 'Single Sided Average Areal Loading (EMD mg/cm2)',
    'final_thickness_microns': 'Final Thickness (Microns)',
    'density_before_g_cm3': 'Density Before (g/cm3)',
    'density_after_g_cm3': 'Density After (g/cm3)',
    'resistance_4pt_avg': 'Average Electrode Resistance -4 Pt Probe (Ohm)',
    'through_plane_res_ohm': 'Through Plain Resistance Measurement',
}

PERF_METRICS = {
    'DChg. Energy(Wh)': 'dchg_energy_wh',
    'Oneset Volt.(V)': 'onset_v',
    'DCIR': 'dcir_mohm',
}

def clean_columns(cols):
    cleaned = []
    for c in cols:
        if pd.isna(c):
            cleaned.append('')
            continue
        c_str = re.sub(r'\s+', ' ', str(c)).strip()
        cleaned.append(c_str)
    return cleaned

@st.cache_data(show_spinner=False)
def load_data(src):
    df_raw = pd.read_excel(src)
    df_raw.columns = clean_columns(df_raw.columns)
    df = df_raw.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df_raw.columns})
    df = df.dropna(how='all')
    if 'source' in df.columns:
        df['source'] = df['source'].astype(str).str.strip()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df_ca = df[df['source'].str.upper() == 'CA'].copy()
    df_lfc = df[df['source'].str.upper() == 'LFC'].copy()
    return df, df_ca, df_lfc

def build_lfc_map(df_ordered):
    mapping = {}
    current_block = []
    for _, row in df_ordered.iterrows():
        src = str(row.get('source', '')).strip().upper()
        name = row.get('electrode_name')
        if src == 'CA':
            current_block.append(name)
        elif src == 'LFC':
            mapping[name] = current_block.copy()
            current_block = []
    return mapping

def summarize(df, cols):
    stats = df[cols].agg(['count', 'mean', 'std', 'median', 'min', 'max']).T
    stats['p10'] = df[cols].quantile(0.10)
    stats['p90'] = df[cols].quantile(0.90)
    return stats

def invert_map(lfc_map):
    inv = {}
    for lfc, electrodes in lfc_map.items():
        for e in electrodes:
            inv[e] = lfc
    return inv

def choose_nbins(n_rows: int) -> int:
    return max(5, min(15, int(np.sqrt(max(n_rows, 1))) + 3))

def aggregate_electrodes(df_ca, lfc_map):
    records = []
    metrics = list(METRIC_LABELS.keys())
    for lfc, electrodes in lfc_map.items():
        subset = df_ca[df_ca['electrode_name'].isin(electrodes)]
        if subset.empty:
            continue
        rec = {'lfc_name': lfc, 'n_electrodes': len(subset)}
        for m in metrics:
            if m in subset.columns:
                rec[f'{m}_mean'] = subset[m].mean()
                rec[f'{m}_std'] = subset[m].std()
                rec[f'{m}_min'] = subset[m].min()
                rec[f'{m}_max'] = subset[m].max()
        records.append(rec)
    return pd.DataFrame(records)

@st.cache_data(show_spinner=False)
def load_performance(lfc_names):
    rows = []
    for lfc in lfc_names:
        path = Path(f'{lfc}.xlsx')
        if not path.exists():
            continue
        try:
            step_df = pd.read_excel(path, sheet_name='step')
        except Exception:
            try:
                step_df = pd.read_excel(path, sheet_name='Step')
            except Exception:
                continue
        step_df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in step_df.columns]
        dcir_col = None
        for c in step_df.columns:
            if 'DCIR' in c.upper():
                dcir_col = c
                break
        col_map = {}
        for raw, norm in PERF_METRICS.items():
            if 'DCIR' in raw:
                if dcir_col:
                    col_map[dcir_col] = norm
            else:
                if raw in step_df.columns:
                    col_map[raw] = norm
        if not col_map:
            continue
        step_df = step_df.rename(columns=col_map)
        mask = (step_df.get('Cycle Index') == 1) & (step_df.get('Step Type').astype(str).str.strip().str.upper() == 'CC DCHG')
        sub = step_df.loc[mask].copy()
        if sub.empty:
            continue
        row = {'lfc_name': lfc}
        for norm in col_map.values():
            if norm in sub.columns:
                row[norm] = pd.to_numeric(sub[norm], errors='coerce').iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)

def filter_data(df_ca, lfc_choice, lfc_map, electrode_to_lfc, date_range):
    df_f = df_ca.copy()
    if date_range:
        allowed_dates = pd.to_datetime(date_range)
        df_f = df_f[df_f['date'].isin(allowed_dates)]
    if lfc_choice and 'All electrodes (CA)' not in lfc_choice:
        allowed = set()
        for lfc in lfc_choice:
            allowed.update(lfc_map.get(lfc, []))
        df_f = df_f[df_f['electrode_name'].isin(allowed)]
    df_f = df_f.copy()
    df_f['lfc_group'] = df_f['electrode_name'].map(electrode_to_lfc).fillna('Unassigned')
    return df_f

def main():
    st.set_page_config(page_title='LFC Quality Dashboard', layout='wide')
    st.title('LFC Quality Dashboard (Electrode-level)')

    st.sidebar.header('Data')
    uploaded = st.sidebar.file_uploader('Upload updated Excel', type=['xlsx'])
    if st.sidebar.button('Refresh data'):
        load_data.clear()
        st.experimental_rerun()

    data_src = uploaded if uploaded is not None else DATA_PATH
    if uploaded is None and not DATA_PATH.exists():
        st.error(f'Missing data file: {DATA_PATH}')
        st.stop()

    df, df_ca, df_lfc = load_data(data_src)
    lfc_to_ca = build_lfc_map(df)
    electrode_to_lfc = invert_map(lfc_to_ca)
    perf_df = load_performance(list(lfc_to_ca.keys()))
    agg_df = aggregate_electrodes(df_ca, lfc_to_ca)
    perf_merged = perf_df.merge(agg_df, on='lfc_name', how='left')

    st.sidebar.header('Filters')
    dates = pd.to_datetime(df_ca['date'].dropna().sort_values().unique())
    if len(dates) == 0:
        st.warning('No date values available in CA data.')
        st.stop()
    date_options = list(dates)
    date_range = st.sidebar.multiselect('Dates', options=date_options, default=date_options, format_func=lambda d: d.strftime('%Y-%m-%d'))

    lfc_options = ['All electrodes (CA)'] + list(lfc_to_ca.keys())
    lfc_choice = st.sidebar.multiselect('LFC (cell) to filter electrodes', options=lfc_options, default=['All electrodes (CA)'])

    tab_elec, tab_perf = st.tabs(['Electrodes', 'Cells (LFC)'])

    with tab_elec:
        df_f = filter_data(df_ca, lfc_choice, lfc_to_ca, electrode_to_lfc, date_range)
        st.caption(f"Filtered electrodes: {len(df_f)} of {len(df_ca)} total")

        if df_f.empty:
            st.warning('No data after filters.')
        else:
            available_metrics = [c for c in METRIC_LABELS if c in df_f.columns]
            stats = summarize(df_f, available_metrics)
            stats = stats.rename(index=METRIC_LABELS)
            st.subheader('Summary statistics')
            st.dataframe(stats.style.format('{:.3f}'))

            st.subheader('Distributions')
            color_col = None if ('All electrodes (CA)' in lfc_choice) else 'lfc_group'
            nbins = choose_nbins(len(df_f))
            for col in available_metrics:
                label = METRIC_LABELS[col]
                if color_col:
                    fig = px.histogram(df_f, x=col, nbins=nbins, marginal='box', title=label, color=color_col, barmode='overlay', opacity=0.65)
                    fig.update_layout(height=400, legend_title_text='LFC')
                else:
                    fig = px.histogram(df_f, x=col, nbins=nbins, marginal='box', title=label, opacity=0.75)
                    fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader('Run charts')
            df_time = df_f.dropna(subset=['date']).sort_values('date')
            if df_time.empty:
                st.info('No dated records available for run charts.')
            else:
                date_order = pd.to_datetime(df_time['date']).dt.normalize().sort_values().unique()
                date_pos_map = {d: i for i, d in enumerate(date_order)}
                lfcs = df_time['lfc_group'].unique()
                offsets = np.linspace(-0.15, 0.15, num=len(lfcs)) if len(lfcs) > 1 else np.array([0.0])
                offset_map = {lfc: offsets[i] for i, lfc in enumerate(lfcs)}

                df_time_plot = df_time.copy()
                df_time_plot['date_norm'] = pd.to_datetime(df_time_plot['date']).dt.normalize()
                df_time_plot['date_pos'] = df_time_plot['date_norm'].map(date_pos_map)
                df_time_plot['lfc_offset'] = df_time_plot['lfc_group'].map(offset_map).fillna(0)
                df_time_plot['date_pos_jitter'] = df_time_plot['date_pos'] + df_time_plot['lfc_offset']

                for col in available_metrics:
                    label = METRIC_LABELS[col]
                    if color_col:
                        fig = px.line(df_time_plot, x='date_pos_jitter', y=col, color=color_col, markers=True, title=f'{label} over time')
                    else:
                        fig = px.scatter(df_time_plot, x='date_pos_jitter', y=col, title=f'{label} over time')
                    fig.update_layout(height=400, legend_title_text='LFC' if color_col else None, xaxis=dict(tickmode='array', tickvals=list(date_pos_map.values()), ticktext=[pd.to_datetime(d).strftime('%Y-%m-%d') for d in date_order], title='Date'))
                    st.plotly_chart(fig, use_container_width=True)

        with st.expander('LFC to electrode mapping (for reference)'):
            for lfc, electrodes in lfc_to_ca.items():
                st.write(f"{lfc}: {len(electrodes)} electrodes")
                st.caption(', '.join(str(e) for e in electrodes))

    with tab_perf:
        st.subheader('Cell (LFC) performance vs electrode characteristics')
        lfcs_selected = list(lfc_to_ca.keys()) if ('All electrodes (CA)' in lfc_choice) else [lfc for lfc in lfc_choice if lfc in lfc_to_ca]
        perf_view = perf_merged.copy()
        if lfcs_selected:
            perf_view = perf_view[perf_view['lfc_name'].isin(lfcs_selected)]
        if perf_view.empty:
            st.warning('No LFC performance data available for the selected filters.')
        else:
            display_cols = ['lfc_name', 'dchg_energy_wh', 'onset_v', 'dcir_mohm']
            for base in ['active_material_g', 'areal_loading_mg_cm2', 'final_thickness_microns', 'density_after_g_cm3', 'resistance_4pt_avg']:
                if f'{base}_mean' in perf_view.columns:
                    display_cols.append(f'{base}_mean')
            perf_table = perf_view[display_cols].rename(columns={
                'lfc_name': 'LFC',
                'dchg_energy_wh': 'DChg Energy (Wh)',
                'onset_v': 'Onset Volt (V)',
                'dcir_mohm': 'DCIR (mOhm)'
            })
            num_fmt = {col: '{:.3f}' for col in perf_table.select_dtypes(include=['number']).columns}
            st.dataframe(perf_table.style.format(num_fmt, na_rep=''))

            perf_targets = [('dchg_energy_wh', 'Discharge Energy (Wh)'), ('onset_v', 'Onset Volt (V)'), ('dcir_mohm', 'DCIR (mOhm)')]
            predictors = [
                ('areal_loading_mg_cm2_mean', 'Areal Loading (mg/cm2, mean)'),
                ('final_thickness_microns_mean', 'Final Thickness (um, mean)'),
                ('density_after_g_cm3_mean', 'Density After (g/cm3, mean)'),
                ('resistance_4pt_avg_mean', '4-pt Resistance (Ohm, mean)'),
                ('active_material_g_mean', 'Active Material (g, mean)')
            ]
            for perf_col, perf_label in perf_targets:
                if perf_col not in perf_view.columns:
                    continue
                for pred_col, pred_label in predictors:
                    if pred_col not in perf_view.columns:
                        continue
                    fig = px.scatter(perf_view, x=pred_col, y=perf_col, text='lfc_name', title=f'{perf_label} vs {pred_label}')
                    fig.update_traces(textposition='top center', marker=dict(size=10, opacity=0.8))
                    fig.update_layout(height=400, xaxis_title=pred_label, yaxis_title=perf_label)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
