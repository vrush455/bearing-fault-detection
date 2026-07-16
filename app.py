from flask import Flask, render_template
import plotly
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
app = Flask(__name__)

# Load your data
results_df = pd.read_csv('online_learning_results.csv')
anomaly_scores = results_df['anomaly_score'].values
times = results_df['time_hours'].values
stages = results_df['stage'].values

@app.route('/')
def index():
    # Create Plot 1: Anomaly Score Over Time
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=times,
        y=anomaly_scores,
        mode='lines',
        name='Anomaly Score',
        line=dict(color='blue', width=2.5)
    ))
    
    # Add stage transition lines
    fig1.add_vline(x=109, line_dash="dash", line_color="orange", 
                   annotation_text="Normal → Early Fault", annotation_position="top right")
    fig1.add_vline(x=218, line_dash="dash", line_color="red",
                   annotation_text="Early → Late Fault", annotation_position="top right")
    
    fig1.update_layout(
        title='🔴 Real-Time Online Learning: Bearing Degradation Detection',
        xaxis_title='Time (hours)',
        yaxis_title='Anomaly Score',
        hovermode='x unified',
        plot_bgcolor='rgba(240,240,240,0.5)',
        template='plotly_white',
        height=500
    )
    
    plot1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create Plot 2: Colored by Stage
    stage_colors = {'Normal': 'green', 'Early_Fault': 'orange', 'Late_Fault': 'red'}
    colors = [stage_colors[stage] for stage in stages]
    
    fig2 = go.Figure()
    
    for stage in ['Normal', 'Early_Fault', 'Late_Fault']:
        mask = np.array(stages) == stage
        fig2.add_trace(go.Scatter(
            x=times[mask],
            y=np.array(anomaly_scores)[mask],
            mode='markers',
            name=stage,
            marker=dict(size=8, color=stage_colors[stage], 
                       line=dict(color='black', width=0.5)),
            hovertemplate=f'<b>{stage}</b><br>Time: %{{x:.1f}}h<br>Score: %{{y:.4f}}<extra></extra>'
        ))
    
    fig2.update_layout(
        title='Bearing Health Progression: 🟢 Healthy → 🔴 Failed',
        xaxis_title='Time (hours)',
        yaxis_title='Anomaly Score',
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)',
        template='plotly_white',
        height=500
    )
    
    plot2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Create summary stats
    normal_avg = np.mean(np.array(anomaly_scores)[np.array(stages) == 'Normal'])
    early_avg = np.mean(np.array(anomaly_scores)[np.array(stages) == 'Early_Fault'])
    late_avg = np.mean(np.array(anomaly_scores)[np.array(stages) == 'Late_Fault'])
    
    stats = {
        'normal_avg': f'{normal_avg:.4f}',
        'early_avg': f'{early_avg:.4f}',
        'late_avg': f'{late_avg:.4f}',
        'total_samples': len(anomaly_scores),
        'total_time': f'{times[-1]:.1f} hours'
    }
    
    return render_template('index.html', 
                         plot1=plot1_json, 
                         plot2=plot2_json,
                         stats=stats)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Bearing Fault Detection Web Dashboard")
    print("="*60)
    print("\n📊 Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, port=5000)