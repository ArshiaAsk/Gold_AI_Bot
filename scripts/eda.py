import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def perform_eda(input_path):
    print(f"--- Starting Exploratory Data Analysis (EDA) on {input_path} ---")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index for better plotting
    df.set_index('Date', inplace=True)
    
    print("\n1. Data Overview:")
    print(df.describe().apply(lambda s: s.apply('{0:.2f}'.format))) 

    # ---------------------------------------------------------
    # Visual 1: Normalized Trend Comparison 
    # ---------------------------------------------------------
    df_normalized = df / df.iloc[0] 
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_normalized.index, df_normalized['Gold_Toman'], label='Gold 18k (Toman)', linewidth=2, color='gold')
    plt.plot(df_normalized.index, df_normalized['USD_Toman'], label='USD (Toman)', linewidth=1.5, color='green', linestyle='--')
    plt.plot(df_normalized.index, df_normalized['Ounce_Toman'], label='Global Ounce (Toman)', linewidth=1.5, color='blue', alpha=0.6)
    plt.plot(df_normalized.index, df_normalized['Oil_Toman'], label='Oil (Toman)', linewidth=1.5, color='black', alpha=0.4)
    
    plt.title('Relative Growth of Assets Over Time (Normalized to Start)', fontsize=16)
    plt.legend()
    plt.ylabel('Growth Factor (1.0 = Start)')
    plt.show()

    # ---------------------------------------------------------
    # Visual 2: Correlation Heatmap 
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix: Which features impact Gold Price most?', fontsize=16)
    plt.show()
    
    print("\n💡 Insight from Correlation:")
    print("If Correlation is > 0.9: Extremely strong relationship (Good for prediction).")
    print("If Correlation is < 0.1: No linear relationship.")

    # ---------------------------------------------------------
    # Visual 3: Scatter Plots 
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gold vs USD
    sns.regplot(x=df['USD_Toman'], y=df['Gold_Toman'], ax=axes[0], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[0].set_title('Gold vs USD (Toman)')
    
    # Gold vs Ounce
    sns.regplot(x=df['Ounce_Toman'], y=df['Gold_Toman'], ax=axes[1], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[1].set_title('Gold vs Ounce (Toman)')
    
    # Gold vs Oil
    sns.regplot(x=df['Oil_Toman'], y=df['Gold_Toman'], ax=axes[2], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[2].set_title('Gold vs Oil (Toman)')
    
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Visual 4: Recent History (30 Days Zoom)
    # ---------------------------------------------------------
    last_30_days = df.head(30)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))

    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Gold 18k (Toman)', color=color)
    ax1.plot(last_30_days.index, last_30_days['Gold_Toman'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()   
    color = 'tab:green'
    ax2.set_ylabel('USD (Toman)', color=color)
    ax2.plot(last_30_days.index, last_30_days['USD_Toman'], color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Zoomed In: Last 30 Days (Gold vs USD)', fontsize=16)
    plt.show()

if __name__ == "__main__":
    perform_eda("final_features_toman.csv")
