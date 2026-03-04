import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, csv_path='Session_Data.csv'):
        self.csv_path = csv_path
        self.output_image = 'Focus_Trends.png'
        self.output_report = 'Final_Analytics_Report.txt'

    def generate(self):
        """
        Processes session logs to create visual and textual analytics.
        """
        if not os.path.exists(self.csv_path):
            return "Error: No session data available to generate report."

        # 1. Load Data
        df = pd.read_csv(self.csv_path)
        
        # 2. Statistical Analysis
        avg_focus = df['focusScore'].mean()
        peak_focus = df['focusScore'].max()
        # Filter for actual recovery events (FRT > 0)
        recovery_events = df[df['recoveryLatency'] > 0]['recoveryLatency']
        avg_frt = recovery_events.mean() if not recovery_events.empty else 0.0
        
        # 3. Generate Focus Trend Visualization
        plt.figure(figsize=(10, 5), facecolor='#1e1e1e')
        ax = plt.axes()
        ax.set_facecolor('#1e1e1e')
        
        plt.plot(df['timestamp'], df['focusScore'], color='#00ff00', linewidth=2, label='Focus Score')
        plt.fill_between(df['timestamp'], df['focusScore'], color='#00ff00', alpha=0.1)
        
        plt.title('CogniFlow: Cognitive Performance Trend', color='white', fontsize=14)
        plt.xlabel('Session Time', color='gray')
        plt.ylabel('Focus Percentage', color='gray')
        plt.ylim(0, 110)
        
        # Styling the chart to match your Glassmorphism UI
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333333')
            
        plt.savefig(self.output_image, dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Generate Textual Summary
        summary = (
            f"--- COGNIFLOW SESSION SUMMARY ---\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Average Focus Score: {avg_focus:.2f}%\n"
            f"Peak Focus Reached: {peak_focus:.2f}%\n"
            f"Avg Focus Recovery Time (FRT): {avg_frt:.2f} seconds\n"
            f"Total Data Points: {len(df)}\n"
            f"Result: {'High Productivity' if avg_focus > 75 else 'Moderate Focus'}\n"
            f"--------------------------------"
        )
        
        with open(self.output_report, 'w') as f:
            f.write(summary)
            
        return f"Report Generated: {self.output_image} and {self.output_report}"

if __name__ == "__main__":
    # Test run
    gen = ReportGenerator()
    print(gen.generate())