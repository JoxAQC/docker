import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.genetic_algorithm import run_genetic_algorithm
from utils.kubernetes_simulator import evaluate_configuration
from utils.analysis_functions import (
    compare_strategies,
    plot_strategy_comparison,
    sensitivity_analysis,
    plot_sensitivity_results,
    plot_cost_breakdown,
    plot_3d_search_space
)

# Configuración de página
st.set_page_config(page_title="Optimizador Avanzado de Kubernetes", layout="wide", page_icon="⚙️")

# Título y descripción
st.title("⚙️ Optimización Avanzada de Configuraciones Kubernetes")
st.markdown("""
Esta herramienta utiliza algoritmos genéticos para encontrar la configuración óptima de recursos 
para tus despliegues en Kubernetes, balanceando costo, performance y cumplimiento de SLA.
""")

# Sidebar con parámetros
with st.sidebar:
    st.header("🔧 Parámetros del Algoritmo")
    population_size = st.slider("Tamaño de población", 10, 200, 50)
    generations = st.slider("Número de generaciones", 5, 150, 30)
    crossover_prob = st.slider("Probabilidad de crossover", 0.1, 1.0, 0.8, 0.05)
    mutation_prob = st.slider("Probabilidad de mutación", 0.01, 0.5, 0.15, 0.01)
    
    st.header("📊 Restricciones del SLA")
    min_cpu = st.slider("CPU mínima (cores)", 0.1, 64.0, 2.0, 0.1)
    max_cpu = st.slider("CPU máxima (cores)", 0.1, 64.0, 16.0, 0.1)
    min_memory = st.slider("Memoria mínima (GB)", 0.1, 128.0, 4.0, 0.1)
    max_memory = st.slider("Memoria máxima (GB)", 0.1, 128.0, 32.0, 0.1)
    min_replicas = st.slider("Mínimo de réplicas", 1, 50, 2)
    max_replicas = st.slider("Máximo de réplicas", 1, 100, 10)
    
    workload = st.selectbox("Escenario de carga", ["Baja", "Media", "Alta"], index=1)
    st.markdown("---")
    st.info("ℹ️ Ajusta los parámetros y haz clic en 'Ejecutar Optimización' para comenzar.")

# Ejecutar optimización
if st.button("🚀 Ejecutar Optimización", type="primary"):
    with st.spinner("🔍 Buscando la configuración óptima..."):
        best_config, history, pop = run_genetic_algorithm(
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            cpu_bounds=(min_cpu, max_cpu),
            memory_bounds=(min_memory, max_memory),
            replicas_bounds=(min_replicas, max_replicas),
            workload=workload
        )
    
    st.success("✅ ¡Optimización completada!")
    st.balloons()
    
    # Evaluar la mejor configuración
    best_eval = evaluate_configuration(best_config, workload)
    
    # Mostrar mejor configuración
    st.subheader("🎯 Mejor Configuración Encontrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU (cores)", f"{best_config['cpu']:.2f}", 
               help="Cores de CPU asignados por pod")
    col2.metric("Memoria (GB)", f"{best_config['memory']:.2f}", 
               help="Memoria asignada por pod")
    col3.metric("Réplicas", best_config['replicas'], 
               help="Número de pods en el clúster")
    
    # Métricas clave
    st.subheader("📈 Métricas Clave")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Costo Mensual", f"${best_eval['cost']:,.2f}", 
                help="Costo total estimado por mes")
    mcol2.metric("Performance", f"{best_eval['performance']*100:.1f}%", 
                "Cumple SLA ✅" if best_eval['sla_compliance'] else "No cumple SLA ❌",
                help="Porcentaje de cumplimiento del SLA")
    mcol3.metric("Eficiencia", f"{best_eval['efficiency_score']:.2f}", 
                help="Puntaje de eficiencia (mayor es mejor)")
    mcol4.metric("Costo/Performance", f"{best_eval['cost_performance_ratio']:.2f}", 
                help="Ratio costo-performance (menor es mejor)")
    
    # Gráfico de desglose de costos
    st.plotly_chart(plot_cost_breakdown(best_eval), use_container_width=True)
    
    # Evolución del algoritmo
    st.subheader("📊 Evolución del Algoritmo Genético")
    df_history = pd.DataFrame(history)
    
    tab1, tab2, tab3 = st.tabs(["Fitness", "Parámetros", "Datos Completos"])
    
    with tab1:
        fig_fitness = px.line(df_history, 
                            x='generation', 
                            y=['best_fitness', 'avg_fitness'],
                            labels={'value': 'Fitness', 'generation': 'Generación'},
                            title="Evolución del Fitness por Generación",
                            color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig_fitness, use_container_width=True)
    
    with tab2:
        fig_params = make_subplots(rows=1, cols=3, subplot_titles=("CPU", "Memoria", "Réplicas"))
        
        fig_params.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['best_cpu'], name='CPU'),
            row=1, col=1
        )
        fig_params.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['best_memory'], name='Memoria'),
            row=1, col=2
        )
        fig_params.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['best_replicas'], name='Réplicas'),
            row=1, col=3
        )
        
        fig_params.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_params, use_container_width=True)
    
    with tab3:
        st.dataframe(df_history.style
                    .background_gradient(subset=['best_fitness'], cmap='Blues')
                    .format({'best_cpu': '{:.2f}', 'best_memory': '{:.2f}'}))
    
    # Comparación con otras estrategias
    st.subheader("🆚 Comparación con Otras Estrategias")
    bounds = {
        'cpu': (min_cpu, max_cpu),
        'memory': (min_memory, max_memory),
        'replicas': (min_replicas, max_replicas)
    }
    df_comparison = compare_strategies(best_config, workload, bounds)
    st.plotly_chart(plot_strategy_comparison(df_comparison), use_container_width=True)
    
    # Mostrar tabla comparativa con formato
    st.dataframe(df_comparison.style
                .format({
                    'Costo ($/mes)': '${:,.2f}',
                    'Performance': '{:.2%}',
                    'Costo/Performance': '{:.2f}',
                    'Eficiencia': '{:.2f}'
                })
                .applymap(lambda x: 'color: green' if x == '✅' else 'color: red', 
                         subset=['Cumple SLA']))
    
    # Análisis de sensibilidad
    st.subheader("📌 Análisis de Sensibilidad")
    df_sensitivity = sensitivity_analysis(best_config, workload)
    st.plotly_chart(plot_sensitivity_results(df_sensitivity), use_container_width=True)
    
    # Visualización 3D del espacio de búsqueda
    st.subheader("🌌 Espacio de Búsqueda 3D")
    st.plotly_chart(plot_3d_search_space(pop, workload, bounds), 
                   use_container_width=True, height=800)

# Sección de ayuda
with st.expander("ℹ️ ¿Cómo interpretar los resultados?"):
    st.markdown("""
    ### Guía de Interpretación
    
    **1. Configuración Óptima**  
    - La mejor combinación de CPU, memoria y réplicas encontrada por el algoritmo.
    - Considera tanto el costo como el cumplimiento del SLA.
    
    **2. Métricas Clave**  
    - **Costo Mensual**: Estimación del costo total de la configuración.
    - **Performance**: Porcentaje de cumplimiento del SLA (95%+ es ideal).
    - **Eficiencia**: Balance entre utilización de recursos y costo.
    - **Costo/Performance**: Ratio que debe minimizarse.
    
    **3. Comparación de Estrategias**  
    - Compara la solución óptima contra enfoques comunes:
      - **Eficiencia en Costo**: Minimiza recursos (puede violar SLA).
      - **Alta Disponibilidad**: Maximiza recursos (costoso).
      - **Balanceado**: Punto medio entre ambos extremos.
    
    **4. Análisis de Sensibilidad**  
    - Muestra cómo cambian las métricas al variar cada parámetro.
    - Ayuda a entender la robustez de la configuración óptima.
    
    **5. Espacio de Búsqueda 3D**  
    - Visualización interactiva de cómo diferentes configuraciones afectan el fitness.
    - El tamaño de los puntos representa la performance.
    - El color representa el fitness (más oscuro = mejor).
    """)

# Notas finales
st.markdown("---")
st.caption("""
*Nota: Los costos son estimaciones basadas en precios promedio de cloud providers.  
El modelo de performance simula cargas de trabajo típicas pero puede necesitar ajustes para casos específicos.*
""")