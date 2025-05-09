
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score

# Configuración de página
st.set_page_config(
    page_title="Tokyo Airbnb Explorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos personalizados
with open('style.css', 'w') as f:
    f.write("""
        .dashboard-title {
            color: #e91e63;
            text-align: center;
            padding: 10px;
            font-size: 80px;
        }
        .dashboard-subtitle {
            color: #666666;
            text-align: center;
            padding-bottom: 20px;
            font-size: 24px;
        }
        .metric-card {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            height: 200px;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #e91e63;
        }
        .metric-label {
            font-size: 16px;
            color: #666666;
        }
    """)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Función para cargar datos
@st.cache_resource
def load_data():
    df = pd.read_csv("Tokio_limpio.csv")
    
    # Limpieza de datos (suponiendo formato similar a los ejemplos)
    # Convertir precio de string a float (eliminar símbolos y convertir)
    if 'price' in df.columns:
        df['price'] = df['price'].replace('[$,]', '', regex=True).astype(float)
    
    # Convertir porcentajes si existen
    for col in ['host_response_rate', 'host_acceptance_rate']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip('%').replace('Without information', pd.NA).astype(float) / 100
    
    # Seleccionar columnas numéricas y categóricas
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = numeric_df.columns.tolist()
    
    text_df = df.select_dtypes(include=['object', 'bool'])
    text_cols = text_df.columns.tolist()
    
    # Crear lista de vecindarios únicos
    if 'neighbourhood_cleansed' in df.columns:
        neighborhoods = df['neighbourhood_cleansed'].dropna().unique()
    else:
        neighborhoods = []
    
    # Crear lista de tipos de propiedades únicas
    if 'property_type' in df.columns:
        property_types = df['property_type'].dropna().unique()
    else:
        property_types = []
    
    # Crear lista de tipos de habitaciones únicas
    if 'room_type' in df.columns:
        room_types = df['room_type'].dropna().unique()
    else:
        room_types = []
    
    return df, numeric_cols, text_cols, neighborhoods, property_types, room_types

# Cargar datos
df, numeric_cols, text_cols, neighborhoods, property_types, room_types = load_data()

# Función para la página principal
def main():
    # Título del dashboard con logo de Tokio
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center;">
            <h1 class="dashboard-title">TOKYO AIRBNB EXPLORER</h1>
        </div>
        <p class="dashboard-subtitle">Análisis interactivo del mercado de alojamientos en Tokio</p>
    """, unsafe_allow_html=True)
    
    # Mostrar imagen de Tokio
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center;">
                <img src="https://www.travelbook.de/data/uploads/2022/11/gettyimages-1284581217-1-1040x690.jpg" width="600">
            </div>
        """, unsafe_allow_html=True)
    
    # Métricas clave del dataset
    st.markdown("### Métricas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total de Propiedades</div>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_price = df['price'].mean() if 'price' in df.columns else 0
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">${:.0f}</div>
                <div class="metric-label">Precio Promedio</div>
            </div>
        """.format(avg_price), unsafe_allow_html=True)
    
    with col3:
        if 'room_type' in df.columns:
            top_room = df['room_type'].value_counts().idxmax()
        else:
            top_room = "N/A"
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Tipo de Habitación Más Común</div>
            </div>
        """.format(top_room), unsafe_allow_html=True)
    
    with col4:
        if 'review_scores_rating' in df.columns:
            avg_rating = df['review_scores_rating'].mean()
        else:
            avg_rating = 0
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}</div>
                <div class="metric-label">Calificación Promedio</div>
            </div>
        """.format(avg_rating), unsafe_allow_html=True)
    
    # Muestra los datos
    with st.expander("Ver datos completos"):
        st.dataframe(df)

# Función para el análisis de precios
def price_analysis():
    st.markdown("## Análisis de Precios")
    
    # Filtros laterales
    st.sidebar.markdown("### Filtros de Precio")
    
    min_price = float(df['price'].min())
    max_price = float(df['price'].max())
    price_range = st.sidebar.slider("Rango de precios ($)", 
                                   min_price, 
                                   max_price, 
                                   (min_price, max_price * 0.8))
    
    # Filtrar por vecindario si está disponible
    if 'neighbourhood_cleansed' in df.columns:
        selected_neighborhoods = st.sidebar.multiselect(
            "Vecindario", 
            options=neighborhoods,
            default=neighborhoods[:5] if len(neighborhoods) > 5 else neighborhoods
        )
    else:
        selected_neighborhoods = []
    
    # Filtrar por tipo de habitación si está disponible
    if 'room_type' in df.columns:
        selected_room_types = st.sidebar.multiselect(
            "Tipo de habitación", 
            options=room_types,
            default=room_types
        )
    else:
        selected_room_types = []
    
    # Aplicar filtros
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & 
                             (filtered_df['price'] <= price_range[1])]
    
    if selected_neighborhoods and 'neighbourhood_cleansed' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['neighbourhood_cleansed'].isin(selected_neighborhoods)]
    
    if selected_room_types and 'room_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['room_type'].isin(selected_room_types)]
    
    # Mostrar gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribución de Precios por Tipo de Habitación")
        if 'room_type' in filtered_df.columns:
            fig = px.box(filtered_df, x="room_type", y="price", 
                         color="room_type", 
                         title="Distribución de Precios por Tipo de Habitación")
            fig.update_layout(xaxis_title="Tipo de Habitación", 
                             yaxis_title="Precio ($)",
                             plot_bgcolor='rgba(0, 0, 0, 0)',
                             paper_bgcolor='rgba(0, 0, 0, 0)')
            st.plotly_chart(fig)
        else:
            st.write("No hay datos de tipo de habitación disponibles")
    
    with col2:
        st.markdown("### Precio Promedio por Vecindario")
        if 'neighbourhood_cleansed' in filtered_df.columns:
            neighborhood_avg = filtered_df.groupby('neighbourhood_cleansed')['price'].mean().reset_index()
            neighborhood_avg = neighborhood_avg.sort_values('price', ascending=False).head(10)
            
            fig = px.bar(neighborhood_avg, x='neighbourhood_cleansed', y='price',
                        color='price',
                        title="Top 5 Vecindarios por Precio Promedio",
                        color_continuous_scale=px.colors.sequential.Reds)
            fig.update_layout(xaxis_title="Vecindario", 
                             yaxis_title="Precio Promedio ($)",
                             xaxis={'categoryorder':'total descending'},
                             plot_bgcolor='rgba(0, 0, 0, 0)',
                             paper_bgcolor='rgba(0, 0, 0, 0)')
            st.plotly_chart(fig)
        else:
            st.write("No hay datos de vecindario disponibles")
    
    # Relación entre precio y otras variables
    st.markdown("### Relación entre precio y otras características")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Variable X", 
                            options=[col for col in numeric_cols if col != 'price'],
                            index=0 if len(numeric_cols) > 1 else 0)
    
    with col2:
        if 'room_type' in filtered_df.columns:
            color_var = st.selectbox("Color por categoría", 
                                    options=[col for col in text_cols if col != x_var],
                                    index=text_cols.index('room_type') if 'room_type' in text_cols else 0)
        else:
            color_var = st.selectbox("Color por categoría", 
                                    options=[col for col in text_cols if col != x_var],
                                    index=0 if text_cols else 0)
    
    # Gráfico de dispersión
    fig = px.scatter(filtered_df, x=x_var, y='price', 
                    color=filtered_df[color_var] if color_var in filtered_df.columns else None,
                    size='accommodates' if 'accommodates' in filtered_df.columns else None,
                    hover_name='id' if 'id' in filtered_df.columns else None,
                    title=f"Relación entre {x_var} y precio")
    fig.update_layout(xaxis_title=x_var, 
                     yaxis_title="Precio ($)",
                     plot_bgcolor='rgba(0, 0, 0, 0)',
                     paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig)
    
    # Opción para exportar datos filtrados
    if st.button('Exportar datos filtrados'):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Descargar como CSV",
            data=csv,
            file_name='tokyo_airbnb_filtered.csv',
            mime='text/csv',
        )

# Función para el análisis de categorías
def category_analysis():
    st.markdown("## Análisis por Categorías")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat_var = st.selectbox("Variable Categórica", 
                              options=[col for col in text_cols],
                              index=text_cols.index('room_type') if 'room_type' in text_cols else 0)
    
    with col2:
        num_var = st.selectbox("Variable Numérica", 
                              options=[col for col in numeric_cols],
                              index=numeric_cols.index('price') if 'price' in numeric_cols else 0)
    
    # Gráfico de torta
    st.markdown("### Distribución por categoría")
    fig = px.pie(df, names=cat_var, values=num_var, 
                title=f"Distribución de {num_var} por {cat_var}",
                color_discrete_sequence=px.colors.sequential.Reds_r)
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',
                     paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig)
    
    # Gráfico de barras
    st.markdown("### Comparativa por categoría")
    grouped = df.groupby(cat_var)[num_var].mean().reset_index()
    grouped = grouped.sort_values(num_var, ascending=False)
    
    fig = px.bar(grouped, x=cat_var, y=num_var, color=cat_var,
                title=f"Promedio de {num_var} por {cat_var}")
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                     xaxis_title=cat_var,
                     yaxis_title=f"Promedio de {num_var}",
                     plot_bgcolor='rgba(0, 0, 0, 0)',
                     paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig)

# Función para el análisis de correlaciones y tendencias
def correlation_analysis():
    st.markdown("## Correlaciones y Tendencias")
    
    # Seleccionar variables para análisis de correlación
    correlation_vars = st.multiselect(
        "Selecciona variables para el análisis de correlación",
        options=numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    )
    
    if correlation_vars:
        # Crear matriz de correlación
        corr_matrix = df[correlation_vars].corr()
        
        # Crear heatmap con seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
        plt.title("Matriz de Correlación")
        st.pyplot(fig)
    
    # Análisis de tendencias
    st.markdown("### Tendencias por variables seleccionadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_x = st.selectbox("Variable X (tendencia)", 
                              options=numeric_cols,
                              index=0)
    
    with col2:
        trend_y = st.selectbox("Variable Y (tendencia)", 
                              options=numeric_cols,
                              index=1 if len(numeric_cols) > 1 else 0)
    
    cat_var = st.selectbox("Agrupar por", 
                          options=text_cols,
                          index=text_cols.index('room_type') if 'room_type' in text_cols else 0)
    
    # Crear gráfico de dispersión personalizado (sin trendline de OLS)
    categories = df[cat_var].unique()
    fig = px.scatter(df, x=trend_x, y=trend_y, 
                    color=df[cat_var],
                    title=f"Relación entre {trend_x} y {trend_y} por {cat_var}")
    
    # Calculamos manualmente las líneas de tendencia usando LinearRegression
    for category in categories:
        df_cat = df[df[cat_var] == category]
        if len(df_cat) > 1:  # Aseguramos tener suficientes puntos
            X = df_cat[trend_x].values.reshape(-1, 1)
            y = df_cat[trend_y].values
            
            try:
                # Crear y entrenar modelo de regresión lineal
                model = LinearRegression()
                model.fit(X, y)
                
                # Crear puntos para la línea de tendencia
                X_line = np.array([[df_cat[trend_x].min()], [df_cat[trend_x].max()]])
                y_line = model.predict(X_line)
                
                # Añadir línea de tendencia al gráfico
                fig.add_scatter(x=X_line.flatten(), y=y_line, 
                               mode='lines', 
                               name=f'Tendencia {category}',
                               line=dict(dash='dash'))
            except:
                # Si hay error en la regresión, continuamos sin añadir línea de tendencia
                pass
    
    fig.update_layout(xaxis_title=trend_x,
                     yaxis_title=trend_y,
                     plot_bgcolor='rgba(0, 0, 0, 0)',
                     paper_bgcolor='rgba(0, 0, 0, 0)')
    st.plotly_chart(fig)

# Función para el análisis predictivo
#####
def predictive_analysis():
    st.markdown("## Análisis Predictivo")
    
    model_type = st.radio("Selecciona el tipo de modelo:",
                         ["Regresión Lineal", "Regresión Logística"])
    
    # Preparación de datos
    if model_type == "Regresión Lineal":
        target_col = st.selectbox("Variable objetivo (Y)",
                                 options=numeric_cols,
                                index=numeric_cols.index('price') if 'price' in numeric_cols else 0)
        
        feature_cols = st.multiselect("Variables predictoras (X)",
                                     options=[col for col in numeric_cols if col != target_col],
                                    default=[col for col in numeric_cols if col != target_col][:3])
        
        if feature_cols:
            X = df[feature_cols].dropna()
            y = df.loc[X.index, target_col]
            
            # Entrenar modelo
            model = LinearRegression()
            model.fit(X, y)
            
            # Predecir y evaluar
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            st.success(f"Coeficiente de determinación R²: {r2:.3f}")
            
            # Visualizar resultados con dos conjuntos de puntos superpuestos
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Puntos azules para valores reales
            ax.scatter(range(len(y)), y, color='blue', alpha=0.6, label='Valores Reales')
            
            # Puntos rojos para predicciones
            ax.scatter(range(len(y)), y_pred, color='red', alpha=0.6, label='Predicciones')
            
            ax.set_xlabel("Índice de Observación")
            ax.set_ylabel(target_col)
            ax.set_title("Valores Reales vs. Predicciones")
            ax.legend()
            st.pyplot(fig)
            
            # Mostrar coeficientes
            coef = pd.DataFrame({'Variable': feature_cols, 'Coeficiente': model.coef_})
            st.markdown("### Coeficientes del modelo")
            st.dataframe(coef)
            
            # Exportar resultados
            if st.button('Exportar resultados del modelo'):
                result_df = pd.DataFrame({'Actual': y, 'Predicción': y_pred})
                result_csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Descargar resultados",
                    data=result_csv,
                    file_name="resultados_regresion_lineal.csv",
                    mime="text/csv"
                )
    #####
    elif model_type == "Regresión Logística":
        # Para regresión logística necesitamos una variable binaria
        if 'host_is_superhost' in df.columns:
            default_target = 'host_is_superhost'
        elif 'instant_bookable' in df.columns:
            default_target = 'instant_bookable'
        else:
            default_target = text_cols[0]
        
        target_col = st.selectbox("Variable objetivo (Y) - debe ser binaria", 
                                 options=text_cols,
                                 index=text_cols.index(default_target) if default_target in text_cols else 0)
        
        feature_cols = st.multiselect("Variables predictoras (X)", 
                                     options=numeric_cols,
                                     default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
        
        if feature_cols:
            # Preparar datos
            df_model = df.copy()
            
            # Convertir variable objetivo a numérica si es categórica
            if df_model[target_col].dtype == object:
                le = LabelEncoder()
                df_model[target_col] = le.fit_transform(df_model[target_col])
            
            X = df_model[feature_cols].dropna()
            y = df_model.loc[X.index, target_col]
            
            # Split de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar modelo
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            st.success(f"Precisión del modelo: {acc:.3f}")
            
            # Mostrar importancia de características
            coef = pd.DataFrame({'Variable': feature_cols, 'Coeficiente': model.coef_[0]})
            coef = coef.sort_values('Coeficiente', ascending=False)
            
            st.markdown("### Importancia de características")
            
            fig = px.bar(coef, x='Variable', y='Coeficiente', 
                        title="Importancia de variables en el modelo",
                        color='Coeficiente',
                        color_continuous_scale=px.colors.sequential.Reds)
            fig.update_layout(xaxis_title="Variable",
                             yaxis_title="Coeficiente",
                             plot_bgcolor='rgba(0, 0, 0, 0)',
                             paper_bgcolor='rgba(0, 0, 0, 0)')
            st.plotly_chart(fig)
            
            # Exportar resultados
            if st.button('Exportar resultados del modelo'):
                result_df = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred})
                result_csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Descargar resultados",
                    data=result_csv,
                    file_name="resultados_regresion_logistica.csv",
                    mime="text/csv"
                )
            
        #######################
        # Añadir opción para matriz de confusión de variables dicotómicas
        st.markdown("### Matriz de Confusión para Variables Dicotómicas")

        # Lista de variables dicotómicas disponibles
        dichotomous_vars = [col for col in df.columns if col in [
            'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 
            'has_availability', 'instant_bookable'
        ]]

        # Verificar si existen variables dicotómicas en el dataset
        if dichotomous_vars:
            var1 = st.selectbox("Variable 1", options=dichotomous_vars)
            var2 = st.selectbox("Variable 2", options=[v for v in dichotomous_vars if v != var1])
            
            # Preparar datos para matriz de confusión
            df_confmat = df[[var1, var2]].copy().dropna()
            
            # Convertir a valores numéricos si son categóricos
            for var in [var1, var2]:
                if df_confmat[var].dtype == object:
                    le = LabelEncoder()
                    df_confmat[var] = le.fit_transform(df_confmat[var])
            
            # Crear matriz de confusión
            conf_data = pd.crosstab(df_confmat[var1], df_confmat[var2], 
                                rownames=[var1], colnames=[var2], 
                                normalize='index')
            
            # Visualizar matriz de confusión
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_data, annot=True, cmap='RdBu_r', fmt='.2f', ax=ax)
            plt.title(f"Matriz de Confusión: {var1} vs {var2}")
            st.pyplot(fig)
            
            # Calcular correlación
            correlation = df_confmat[var1].corr(df_confmat[var2])
            st.write(f"Correlación entre {var1} y {var2}: {correlation:.3f}")
            
            # Interpretación
            if correlation > 0.5:
                st.info("Las variables muestran una correlación positiva fuerte.")
            elif correlation > 0.2:
                st.info("Las variables muestran una correlación positiva moderada.")
            elif correlation > -0.2:
                st.info("Las variables muestran poca o ninguna correlación.")
            elif correlation > -0.5:
                st.info("Las variables muestran una correlación negativa moderada.")
            else:
                st.info("Las variables muestran una correlación negativa fuerte.")
        else:
            st.warning("No se encontraron variables dicotómicas en el dataset.")
            #######################

# Menú principal
st.sidebar.title(" Tokyo Airbnb Explorer")
st.sidebar.markdown("---")

# Selección de vista
view = st.sidebar.radio("Selecciona una vista:", 
                      ["Inicio", 
                       "Análisis de Precios", 
                       "Análisis por Categorías", 
                       "Correlaciones y Tendencias", 
                       "Análisis Predictivo"])



# Mostrar la vista seleccionada
if view == "Inicio":
    main()
elif view == "Análisis de Precios":
    price_analysis()
elif view == "Análisis por Categorías":
    category_analysis()
elif view == "Correlaciones y Tendencias":
    correlation_analysis()
elif view == "Análisis Predictivo":
    predictive_analysis()

# Footer
st.markdown("---")
st.markdown("""
""", unsafe_allow_html=True)
