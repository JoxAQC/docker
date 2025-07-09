import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils.kubernetes_simulator import evaluate_configuration

def compare_strategies(best_config, workload, bounds):
    """Compara la configuración óptima contra estrategias comunes"""
    strategies = {
        'Cost-Efficient': {
            'cpu': bounds['cpu'][0] * 1.1,
            'memory': bounds['memory'][0] * 1.1,
            'replicas': bounds['replicas'][0]
        },
        'High-Availability': {
            'cpu': bounds['cpu'][1] * 0.9,
            'memory': bounds['memory'][1] * 0.9,
            'replicas': bounds['replicas'][1]
        },
        'Balanced': {
            'cpu': np.mean(bounds['cpu']),
            'memory': np.mean(bounds['memory']),
            'replicas': int(np.mean(bounds['replicas']))
        },
        'Optimized (GA)': best_config
    }
    
    comparison = []
    for name, config in strategies.items():
        res = evaluate_configuration(config, workload)
        comparison.append({
            'Estrategia': name,
            'CPU (cores)': config['cpu'],
            'Memoria (GB)': config['memory'],
            'Réplicas': config['replicas'],
            'Costo ($/mes)': res['cost'],
            'Performance': res['performance'],
            'Costo/Performance': res['cost_performance_ratio'],
            'Eficiencia': res['efficiency_score'],
            'Cumple SLA': "✅" if res['sla_compliance'] else "❌"
        })
    
    return pd.DataFrame(comparison)

def plot_strategy_comparison(df_comparison):
    """Subgráficos independientes para cada métrica"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Costo Mensual", "Performance", "Eficiencia", "Costo/Performance"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # Gráfico de Costo
    fig.add_trace(
        go.Bar(
            x=df_comparison['Estrategia'],
            y=df_comparison['Costo ($/mes)'],
            name='Costo',
            marker_color='#636EFA',
            text=[f"${x:,.2f}" for x in df_comparison['Costo ($/mes)']],
            textposition='auto'
        ),
        row=1, col=1
    )

    # Gráfico de Performance
    fig.add_trace(
        go.Bar(
            x=df_comparison['Estrategia'],
            y=df_comparison['Performance'],
            name='Performance',
            marker_color='#EF553B',
            text=[f"{x:.1%}" for x in df_comparison['Performance']],
            textposition='auto'
        ),
        row=1, col=2
    )

    # Gráfico de Eficiencia
    fig.add_trace(
        go.Bar(
            x=df_comparison['Estrategia'],
            y=df_comparison['Eficiencia'],
            name='Eficiencia',
            marker_color='#00CC96',
            text=[f"{x:.2f}" for x in df_comparison['Eficiencia']],
            textposition='auto'
        ),
        row=2, col=1
    )

    # Gráfico de Costo/Performance
    fig.add_trace(
        go.Bar(
            x=df_comparison['Estrategia'],
            y=df_comparison['Costo/Performance'],
            name='Costo/Performance',
            marker_color='#AB63FA',
            text=[f"{x:.2f}" for x in df_comparison['Costo/Performance']],
            textposition='auto'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title_text="Comparación Detallada por Métricas",
        height=800,
        showlegend=False
    )
    
    return fig

def sensitivity_analysis(best_config, workload, variation=0.2):
    """Analiza sensibilidad a cambios en parámetros"""
    base_eval = evaluate_configuration(best_config, workload)
    results = []
    
    # Variar CPU
    for factor in [1-variation, 1, 1+variation]:
        modified = best_config.copy()
        modified['cpu'] = max(0.1, modified['cpu'] * factor)
        res = evaluate_configuration(modified, workload)
        results.append({
            'Parámetro': 'CPU',
            'Variación': f"{factor:.1f}x",
            'Costo': res['cost'],
            'Performance': res['performance'],
            'Δ Costo': res['cost'] - base_eval['cost'],
            'Δ Performance': res['performance'] - base_eval['performance']
        })
    
    # Variar Memoria
    for factor in [1-variation, 1, 1+variation]:
        modified = best_config.copy()
        modified['memory'] = max(0.1, modified['memory'] * factor)
        res = evaluate_configuration(modified, workload)
        results.append({
            'Parámetro': 'Memoria',
            'Variación': f"{factor:.1f}x",
            'Costo': res['cost'],
            'Performance': res['performance'],
            'Δ Costo': res['cost'] - base_eval['cost'],
            'Δ Performance': res['performance'] - base_eval['performance']
        })
    
    # Variar Réplicas
    for factor in [1-variation, 1, 1+variation]:
        modified = best_config.copy()
        modified['replicas'] = max(1, int(modified['replicas'] * factor))
        res = evaluate_configuration(modified, workload)
        results.append({
            'Parámetro': 'Réplicas',
            'Variación': f"{factor:.1f}x",
            'Costo': res['cost'],
            'Performance': res['performance'],
            'Δ Costo': res['cost'] - base_eval['cost'],
            'Δ Performance': res['performance'] - base_eval['performance']
        })
    
    return pd.DataFrame(results)

def plot_sensitivity_results(df_sensitivity):
    """Visualiza resultados del análisis de sensibilidad"""
    fig = px.line(df_sensitivity, 
                 x='Variación', 
                 y=['Δ Costo', 'Δ Performance'],
                 facet_col='Parámetro',
                 title="Impacto de Variaciones en Parámetros",
                 labels={'value': 'Cambio respecto a óptimo', 'Variación': 'Factor de variación'},
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="Cambio en métrica"
    )
    
    # Añadir línea en cero para referencia
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
    
    return fig

def plot_cost_breakdown(eval_result):
    """Gráfico de desglose de costos"""
    breakdown = eval_result['cost_breakdown']
    df = pd.DataFrame({
        'Componente': ['CPU', 'Memoria', 'Réplicas'],
        'Porcentaje': [breakdown['cpu'], breakdown['memory'], breakdown['replicas']]
    })
    
    fig = px.pie(df, values='Porcentaje', names='Componente',
                title="Distribución de Costos",
                color_discrete_sequence=px.colors.sequential.RdBu)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_3d_search_space(population, workload, bounds):
    """Visualización 3D del espacio de búsqueda"""
    # Muestrear espacio de parámetros
    samples = []
    for _ in range(100):
        cpu = np.random.uniform(bounds['cpu'][0], bounds['cpu'][1])
        memory = np.random.uniform(bounds['memory'][0], bounds['memory'][1])
        replicas = np.random.randint(bounds['replicas'][0], bounds['replicas'][1]+1)
        
        config = {'cpu': cpu, 'memory': memory, 'replicas': replicas}
        res = evaluate_configuration(config, workload)
        samples.append({
            'CPU': cpu,
            'Memoria': memory,
            'Réplicas': replicas,
            'Costo': res['cost'],
            'Performance': res['performance'],
            'Fitness': res['cost'] * (1.1 - res['performance'])
        })
    
    df = pd.DataFrame(samples)
    
    # Gráfico 3D
    fig = px.scatter_3d(
        df,
        x='CPU',
        y='Memoria',
        z='Réplicas',
        color='Fitness',
        size='Performance',
        hover_data=['Costo', 'Performance'],
        title="Espacio de Búsqueda 3D",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='CPU (cores)',
            yaxis_title='Memoria (GB)',
            zaxis_title='Réplicas'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig