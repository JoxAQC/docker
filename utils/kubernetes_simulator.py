import numpy as np

def evaluate_configuration(config, workload="Media"):
    """Evalúa una configuración de Kubernetes con métricas avanzadas"""
    # Validar inputs con límites físicos realistas
    cpu = max(0.1, min(64.0, config['cpu']))
    memory = max(0.1, min(128.0, config['memory']))
    replicas = max(1, min(100, config['replicas']))
    
    # Modelo de costos no lineal con economías de escala
    cpu_cost = 0.024 * (cpu ** 1.1) * 1000  # $24/core/mes con ajuste no lineal
    memory_cost = 0.008 * (memory ** 1.05) * 1000  # $8/GB/mes
    replicas_cost = (0.001 * replicas ** 1.2) * 1000  # Costo no lineal por réplicas
    
    total_cost = max(0.1, (cpu_cost + memory_cost) * replicas + replicas_cost)
    
    # Modelo de performance con rendimientos decrecientes
    base_cpu = 10 if workload == "Media" else (5 if workload == "Baja" else 20)
    base_memory = 20 if workload == "Media" else (10 if workload == "Baja" else 40)
    
    # Efectos de escala (más réplicas = menos eficiencia)
    effective_cpu = cpu * (1 - 0.01 * (replicas ** 0.5))
    effective_memory = memory * (1 - 0.005 * (replicas ** 0.7))
    
    cpu_performance = min(1.0, max(0, (effective_cpu * replicas) / (base_cpu * replicas ** 1.05)))
    memory_performance = min(1.0, max(0, (effective_memory * replicas) / (base_memory * replicas ** 1.03)))
    
    # Performance compuesta ponderada
    performance = 0.7 * min(cpu_performance, memory_performance) + 0.3 * ((cpu_performance + memory_performance) / 2)
    
    # Métricas avanzadas
    cost_performance_ratio = total_cost / (performance + 1e-6)
    resource_utilization = (cpu_performance + memory_performance) / 2
    efficiency_score = resource_utilization / (total_cost / 1000 + 1e-6)
    sla_compliance = performance >= 0.95
    
    cost_breakdown = {
        'cpu': (cpu_cost * replicas) / total_cost * 100,
        'memory': (memory_cost * replicas) / total_cost * 100,
        'replicas': replicas_cost / total_cost * 100
    }
    
    return {
        'cost': total_cost,
        'performance': performance,
        'cpu_usage': cpu_performance,
        'memory_usage': memory_performance,
        'cost_performance_ratio': cost_performance_ratio,
        'resource_utilization': resource_utilization,
        'efficiency_score': efficiency_score,
        'sla_compliance': sla_compliance,
        'cost_breakdown': cost_breakdown,
        'config': {
            'cpu': cpu,
            'memory': memory,
            'replicas': replicas
        }
    }