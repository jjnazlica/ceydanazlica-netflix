# =============================================================================
# Imports and Setup
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
from datetime import datetime
import calendar
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose

# =============================================================================
# Visualization Theme Setup
# =============================================================================

# Netflix-inspired color palette
NETFLIX_COLORS = {
    'primary': '#E50914',     # Netflix Red
    'secondary': '#564D4D',   # Dark Gray
    'accent1': '#221F1F',     # Netflix Black
    'accent2': '#F5F5F1',     # Light Gray
    'complementary': ['#00838F', '#2E7D32', '#5E35B1', '#FF6F00', '#C62828']  # For multiple series
}

# Set modern style for all visualizations
plt.style.use('seaborn-v0_8-darkgrid')  # Using specific seaborn style version
sns.set_theme(style="darkgrid")  # Apply dark grid theme
sns.set_palette([NETFLIX_COLORS['primary']] + NETFLIX_COLORS['complementary'])

# Custom style function
def apply_netflix_style(ax):
    """Apply Netflix-inspired styling to an axis"""
    # Set background color
    ax.set_facecolor(NETFLIX_COLORS['accent2'])
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(NETFLIX_COLORS['secondary'])
    ax.spines['bottom'].set_color(NETFLIX_COLORS['secondary'])
    
    # Style ticks and labels
    ax.tick_params(colors=NETFLIX_COLORS['secondary'])
    ax.yaxis.label.set_color(NETFLIX_COLORS['secondary'])
    ax.xaxis.label.set_color(NETFLIX_COLORS['secondary'])
    ax.title.set_color(NETFLIX_COLORS['accent1'])
    ax.title.set_fontweight('bold')
    
    # Grid style
    ax.grid(color=NETFLIX_COLORS['secondary'], alpha=0.1)

# =============================================================================
# Data Preprocessing
# =============================================================================
def preprocess_netflix_data(netflix_data):
    """Initial data preprocessing and feature engineering"""
    # Create a copy of the dataframe to avoid warnings
    df = netflix_data.copy()
    
    # Split Title into Series, Season, and Episode
    split_columns = df['Title'].str.extract(r'^(.*?)(?:: Season (\d+))?: (.*)$')
    split_columns.columns = ['Series', 'Season', 'Episode']
    
    # Fill missing values without using inplace
    split_columns['Series'] = split_columns['Series'].fillna(df['Title'])
    split_columns['Season'] = split_columns['Season'].fillna('Movie')
    split_columns['Episode'] = split_columns['Episode'].fillna('Movie')
    
    # Convert Season to category
    split_columns['Season'] = split_columns['Season'].astype('category')
    
    # Combine with original data
    netflix_data_cleaned = pd.concat([df.drop(columns=['Title']), split_columns], axis=1)
    
    # Suppress the warning by using coerce for invalid dates
    netflix_data_cleaned['Date'] = pd.to_datetime(netflix_data_cleaned['Date'], 
                                                format='mixed',  # Use mixed format
                                                dayfirst=True)   # Assume day comes first
    
    # Add temporal features
    netflix_data_cleaned['Year'] = netflix_data_cleaned['Date'].dt.year
    netflix_data_cleaned['Month'] = netflix_data_cleaned['Date'].dt.month
    netflix_data_cleaned['Day'] = netflix_data_cleaned['Date'].dt.day
    netflix_data_cleaned['Weekday'] = netflix_data_cleaned['Date'].dt.day_name()
    netflix_data_cleaned['Season'] = pd.cut(netflix_data_cleaned['Date'].dt.month, 
                                          bins=[0,3,6,9,12], 
                                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    print("Data preprocessing completed successfully.")
    print(f"Columns in processed data: {', '.join(netflix_data_cleaned.columns)}")
    
    return netflix_data_cleaned

# =============================================================================
# Basic Viewing Analysis
# =============================================================================
def analyze_viewing_trends(df):
    """Analyze viewing patterns over time"""
    
    # Monthly viewing counts
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly_views = df.groupby('YearMonth').size()
    
    # Year-over-year comparison
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    yearly_views = df.groupby('Year').size()
    
    # Weekday patterns
    df['Weekday'] = pd.to_datetime(df['Date']).dt.day_name()
    weekday_views = df.groupby('Weekday').size()
    
    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    monthly_views.plot(kind='line', ax=ax1)
    ax1.set_title('Monthly Viewing Trends')
    ax1.set_ylabel('Number of Views')
    
    yearly_views.plot(kind='bar', ax=ax2)
    ax2.set_title('Yearly Viewing Distribution')
    ax2.set_ylabel('Number of Views')
    
    weekday_views.plot(kind='bar', ax=ax3)
    ax3.set_title('Weekday Viewing Patterns')
    ax3.set_ylabel('Number of Views')
    
    plt.tight_layout()
    return monthly_views, yearly_views, weekday_views

def analyze_series_patterns(df):
    """Detailed analysis of series watching behavior"""
    
    # Most watched series
    series_counts = df[df['Episode'] != 'Movie'].groupby('Series').size().sort_values(ascending=False)
    
    # Average time between episodes for each series - FIXED
    df_series = df[df['Episode'] != 'Movie'].copy()
    df_series['Date'] = pd.to_datetime(df_series['Date'])
    df_series = df_series.sort_values('Date')
    
    series_gaps = {}
    for series in df_series['Series'].unique():
        series_data = df_series[df_series['Series'] == series]
        if len(series_data) > 1:  # Only calculate for series with multiple episodes
            # Calculate gaps for each season separately
            series_data = series_data.sort_values('Date')
            gaps = series_data.groupby('Season', observed=True)['Date'].diff().dropna()
            if not gaps.empty:
                avg_gap = gaps.mean().days if hasattr(gaps.mean(), 'days') else gaps.mean() / pd.Timedelta(days=1)
                if avg_gap > 0:  # Only include positive gaps
                    series_gaps[series] = avg_gap
    
    series_gaps = pd.Series(series_gaps)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    series_counts.head(10).plot(kind='barh', ax=ax1)
    ax1.set_title('Top 10 Most Watched Series')
    
    if not series_gaps.empty:
        series_gaps.sort_values().head(10).plot(kind='barh', ax=ax2)
        ax2.set_title('Average Days Between Episodes (Top 10 Most Frequently Watched)')
    else:
        ax2.text(0.5, 0.5, 'No valid gap data available', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Print detailed statistics
    print("\nDetailed Series Statistics:")
    print(f"Total unique series: {len(series_counts)}")
    if not series_gaps.empty:
        print(f"Average gap between episodes (overall): {series_gaps.mean():.1f} days")
        print("\nTop 5 Series with Shortest Viewing Gaps:")
        print(series_gaps.sort_values().head())
    
    return series_counts, series_gaps

def analyze_content_distribution(df):
    """Analyze movie vs series viewing patterns"""
    
    # Basic counts
    content_type = df['Episode'].apply(lambda x: 'Movie' if x == 'Movie' else 'Series').value_counts()
    
    # Monthly distribution of movies vs series
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
    df['ContentType'] = df['Episode'].apply(lambda x: 'Movie' if x == 'Movie' else 'Series')
    monthly_type_dist = df.pivot_table(
        index='YearMonth', 
        columns='ContentType', 
        aggfunc='size',
        fill_value=0
    )
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    content_type.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
    ax1.set_title('Movie vs Series Distribution')
    
    monthly_type_dist.plot(kind='line', ax=ax2)
    ax2.set_title('Monthly Movie vs Series Viewing')
    ax2.set_ylabel('Number of Views')
    
    plt.tight_layout()
    return content_type, monthly_type_dist

# =============================================================================
# Advanced Analysis Functions
# =============================================================================
def analyze_binge_patterns(df):
    """Detailed binge watching analysis"""
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateOnly'] = df['Date'].dt.date
    
    # Define binge sessions (3+ views per day)
    binge_data = []
    for date, group in df.groupby('DateOnly'):
        if len(group) >= 3:
            binge_data.append({
                'date': date,
                'count': len(group),
                'content': group['Series'].tolist(),
                'type': 'Mixed' if (group['Episode'] == 'Movie').any() else 'Series'
            })
    
    binge_df = pd.DataFrame(binge_data)
    
    # Analyze patterns
    if not binge_df.empty:
        binge_df['month'] = pd.to_datetime(binge_df['date']).dt.month
        monthly_binges = binge_df.groupby('month')['count'].agg(['mean', 'max', 'count'])
        
        # Most common binge series
        all_binge_content = [item for sublist in binge_df['content'] for item in sublist]
        binge_series = pd.Series(all_binge_content).value_counts()
        
        # Visualize
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor(NETFLIX_COLORS['accent2'])
        
        # Average Binge Size
        monthly_binges['mean'].plot(kind='bar', ax=ax1, color=NETFLIX_COLORS['primary'])
        apply_netflix_style(ax1)
        ax1.set_title('Average Binge Size by Month', pad=20, fontsize=12, fontweight='bold')
        
        # Number of Binge Days
        monthly_binges['count'].plot(kind='bar', ax=ax2, color=NETFLIX_COLORS['complementary'][0])
        apply_netflix_style(ax2)
        ax2.set_title('Number of Binge Days by Month', pad=20, fontsize=12, fontweight='bold')
        
        # Most Common Series
        colors = NETFLIX_COLORS['complementary'][:len(binge_series.head(10))]
        binge_series.head(10).plot(kind='barh', ax=ax3, color=colors)
        apply_netflix_style(ax3)
        ax3.set_title('Most Common Series in Binge Sessions', pad=20, fontsize=12, fontweight='bold')
        
        # Distribution
        sns.histplot(data=binge_df['count'], ax=ax4, color=NETFLIX_COLORS['primary'], alpha=0.7)
        apply_netflix_style(ax4)
        ax4.set_title('Distribution of Binge Session Sizes', pad=20, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return monthly_binges, binge_series
    
    return None, None

def analyze_seasonal_patterns(df):
    """Analyze viewing patterns across seasons and holidays"""
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = pd.cut(df['Date'].dt.month, 
                         bins=[0,3,6,9,12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Seasonal viewing patterns
    seasonal_views = df.groupby('Season', observed=True).size()
    
    # Content type by season
    seasonal_content = df.pivot_table(
        index='Season',
        columns=df['Episode'].apply(lambda x: 'Movie' if x == 'Movie' else 'Series'),
        aggfunc='size',
        fill_value=0,
        observed=True
    )
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    seasonal_views.plot(kind='bar', ax=ax1)
    ax1.set_title('Viewing Distribution by Season')
    
    seasonal_content.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_title('Content Type Distribution by Season')
    
    plt.tight_layout()
    return seasonal_views, seasonal_content

def advanced_viewing_analysis(df):
    """Advanced analysis including correlations and viewing patterns"""
    
    # 1. Create additional features for correlation
    df_analysis = df.copy()
    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
    df_analysis['Month'] = df_analysis['Date'].dt.month
    df_analysis['DayOfWeek'] = df_analysis['Date'].dt.dayofweek
    df_analysis['IsWeekend'] = df_analysis['DayOfWeek'].isin([5, 6]).astype(int)
    df_analysis['DaysSinceStart'] = (df_analysis['Date'] - df_analysis['Date'].min()).dt.days
    
    # Count daily views
    daily_views = df_analysis.groupby(df_analysis['Date'].dt.date, observed=True).size()
    df_analysis['DailyViewCount'] = df_analysis['Date'].dt.date.map(daily_views)
    
    # Is it a binge day? (3+ views)
    df_analysis['IsBingeDay'] = (df_analysis['DailyViewCount'] >= 3).astype(int)
    
    # Content type (1 for series, 0 for movies)
    df_analysis['IsSeries'] = (df_analysis['Episode'] != 'Movie').astype(int)
    
    # 2. Create correlation matrix
    correlation_features = [
        'Month', 'DayOfWeek', 'IsWeekend', 'DailyViewCount',
        'IsBingeDay', 'IsSeries', 'DaysSinceStart'
    ]
    
    correlation_matrix = df_analysis[correlation_features].corr()
    
    # 3. Analyze viewing patterns over time
    monthly_series_ratio = df_analysis.groupby('Month', observed=True).agg({
        'IsSeries': 'mean',
        'DailyViewCount': 'mean',
        'IsBingeDay': 'mean'
    })
    
    # 4. Weekend vs Weekday analysis
    weekend_patterns = df_analysis.groupby('IsWeekend', observed=True).agg({
        'IsSeries': ['mean', 'count'],
        'DailyViewCount': 'mean',
        'IsBingeDay': 'mean'
    })
    
    # 5. Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correlation matrix heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Correlation Matrix of Viewing Patterns')
    
    # Monthly patterns
    monthly_series_ratio.plot(ax=ax2)
    ax2.set_title('Monthly Viewing Patterns')
    ax2.set_xlabel('Month')
    
    # Daily view count distribution
    sns.histplot(data=df_analysis, x='DailyViewCount', ax=ax3)
    ax3.set_title('Distribution of Daily View Counts')
    
    # Weekend vs Weekday comparison
    weekend_patterns['IsSeries']['mean'].plot(kind='bar', ax=ax4)
    ax4.set_title('Series Watching: Weekend vs Weekday')
    ax4.set_xticklabels(['Weekday', 'Weekend'])
    
    plt.tight_layout()
    
    # 6. Statistical tests and insights
    print("\nKey Insights:")
    
    # Test if weekend viewing is significantly different
    weekend_vs_weekday = stats.ttest_ind(
        df_analysis[df_analysis['IsWeekend'] == 1]['DailyViewCount'],
        df_analysis[df_analysis['IsWeekend'] == 0]['DailyViewCount']
    )
    print(f"Weekend vs Weekday viewing difference p-value: {weekend_vs_weekday.pvalue:.4f}")
    
    # Binge watching patterns
    print("\nBinge Watching Patterns:")
    print(f"Percentage of binge days: {df_analysis['IsBingeDay'].mean()*100:.1f}%")
    print(f"Average views on binge days: {df_analysis[df_analysis['IsBingeDay']==1]['DailyViewCount'].mean():.1f}")
    
    # Series vs Movies patterns
    print("\nSeries vs Movies:")
    print(f"Series watching percentage: {df_analysis['IsSeries'].mean()*100:.1f}%")
    print(f"Series on weekends: {df_analysis[df_analysis['IsWeekend']==1]['IsSeries'].mean()*100:.1f}%")
    print(f"Series on weekdays: {df_analysis[df_analysis['IsWeekend']==0]['IsSeries'].mean()*100:.1f}%")
    
    return {
        'correlation_matrix': correlation_matrix,
        'monthly_patterns': monthly_series_ratio,
        'weekend_patterns': weekend_patterns,
        'daily_stats': df_analysis.groupby('DayOfWeek')['DailyViewCount'].mean()
    }

def analyze_viewing_trends_advanced(df):
    """Analyze long-term viewing trends and patterns"""
    
    df_trends = df.copy()
    df_trends['Date'] = pd.to_datetime(df_trends['Date'])
    df_trends['YearMonth'] = df_trends['Date'].dt.to_period('M')
    
    # Monthly trends
    monthly_counts = df_trends.groupby('YearMonth').size()
    
    # Calculate moving averages
    rolling_mean = monthly_counts.rolling(window=3).mean()
    
    # Trend analysis
    x = np.arange(len(monthly_counts))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, monthly_counts)
    
    # Seasonality check
    month_seasonality = df_trends.groupby(df_trends['Date'].dt.month).size()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Monthly trend with rolling average
    monthly_counts.plot(ax=ax1, label='Monthly views')
    rolling_mean.plot(ax=ax1, label='3-month moving average')
    ax1.plot(x, slope * x + intercept, '--', label='Trend line')
    ax1.set_title('Viewing Trends Over Time')
    ax1.legend()
    
    # Monthly seasonality
    month_seasonality.plot(kind='bar', ax=ax2)
    ax2.set_title('Monthly Seasonality Pattern')
    ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout()
    
    # Print insights
    print("\nTrend Analysis:")
    print(f"Overall trend slope: {slope:.2f} views per month")
    print(f"R-squared value: {r_value**2:.3f}")
    print(f"Trend p-value: {p_value:.4f}")
    
    return {
        'monthly_counts': monthly_counts,
        'trend_stats': {'slope': slope, 'r_squared': r_value**2, 'p_value': p_value},
        'seasonality': month_seasonality
    }

def analyze_trends_and_seasonality(df):
    """Analyze viewing trends with statistical models and seasonality"""
    
    # Create daily view counts
    daily_views = df.groupby('Date').size().reset_index(name='views')
    daily_views['Date'] = pd.to_datetime(daily_views['Date'])
    daily_views = daily_views.set_index('Date')
    
    # Fill missing dates with 0
    idx = pd.date_range(daily_views.index.min(), daily_views.index.max())
    daily_views = daily_views.reindex(idx, fill_value=0)
    
    # Decompose the time series with a larger period for clearer seasonality
    decomposition = seasonal_decompose(daily_views['views'], period=90)  # Changed from 30 to 90
    
    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(NETFLIX_COLORS['accent2'])
    
    # Original Data and Trend
    ax1.plot(daily_views.index, daily_views['views'], 
            label='Original', color=NETFLIX_COLORS['primary'], alpha=0.3)
    ax1.plot(daily_views.index, decomposition.trend, 
            label='Trend', color=NETFLIX_COLORS['complementary'][0], linewidth=2)
    apply_netflix_style(ax1)
    ax1.set_title('Viewing Trend Analysis', pad=20)
    ax1.legend()
    
    # Simplified Seasonal Pattern
    # Instead of plotting all points, get monthly average seasonal pattern
    seasonal_pattern = pd.DataFrame(decomposition.seasonal)
    seasonal_pattern['month'] = seasonal_pattern.index.month
    monthly_seasonal = seasonal_pattern.groupby('month').mean()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax2.plot(months, monthly_seasonal.values, 
            color=NETFLIX_COLORS['complementary'][1], 
            marker='o', linewidth=2)
    apply_netflix_style(ax2)
    ax2.set_title('Average Monthly Seasonal Pattern', pad=20)
    ax2.tick_params(axis='x', rotation=45)
    
    # Monthly Heatmap
    monthly_views = df.groupby([df['Date'].dt.year, df['Date'].dt.month]).size().unstack()
    monthly_views.columns = months
    sns.heatmap(monthly_views, ax=ax3, cmap='RdYlBu_r', 
                annot=True, fmt='g', cbar_kws={'label': 'Number of Views'})
    apply_netflix_style(ax3)
    ax3.set_title('Monthly Viewing Heatmap', pad=20)
    
    # Model Fit
    X = np.arange(len(daily_views)).reshape(-1, 1)
    y = daily_views['views'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    # Calculate moving average for smoother visualization
    rolling_mean = daily_views['views'].rolling(window=30).mean()
    
    ax4.scatter(X, y, color=NETFLIX_COLORS['primary'], alpha=0.1, s=10)
    ax4.plot(X, rolling_mean.values, color=NETFLIX_COLORS['complementary'][3],
             label='30-day Moving Average', linewidth=2)
    ax4.plot(X, intercept + slope * X, color=NETFLIX_COLORS['complementary'][2],
             label=f'Linear Trend\nR² = {r_value**2:.3f}\np = {p_value:.3f}',
             linewidth=2)
    apply_netflix_style(ax4)
    ax4.set_title('Trend Analysis', pad=20)
    ax4.legend()
    
    plt.tight_layout()
    return decomposition, (slope, intercept, r_value, p_value, std_err)

def analyze_holiday_patterns(df):
    """Analyze viewing patterns during holidays"""
    
    # Get Turkish holidays
    tr_holidays = holidays.TR(years=range(2016, 2025))
    
    # Mark holiday dates
    df['Date'] = pd.to_datetime(df['Date'])
    df['is_holiday'] = df['Date'].apply(lambda x: x in tr_holidays)
    df['holiday_name'] = df['Date'].apply(lambda x: tr_holidays.get(x))
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(NETFLIX_COLORS['accent2'])
    
    # Holiday vs Non-Holiday Viewing
    holiday_views = df.groupby('is_holiday').size()
    holiday_views.plot(kind='bar', ax=ax1, color=[NETFLIX_COLORS['primary'], 
                      NETFLIX_COLORS['complementary'][0]])
    apply_netflix_style(ax1)
    ax1.set_title('Holiday vs Non-Holiday Viewing', pad=20)
    
    # Top Holiday Viewing Days
    holiday_daily = df[df['is_holiday']].groupby(['Date', 'holiday_name']).size()
    top_holidays = holiday_daily.nlargest(10)
    top_holidays.plot(kind='barh', ax=ax2, color=NETFLIX_COLORS['complementary'])
    apply_netflix_style(ax2)
    ax2.set_title('Top Holiday Viewing Days', pad=20)
    
    # Holiday Type Distribution
    holiday_type_dist = df[df['is_holiday']].groupby('holiday_name').size()
    holiday_type_dist.plot(kind='pie', ax=ax3, colors=NETFLIX_COLORS['complementary'])
    apply_netflix_style(ax3)
    ax3.set_title('Distribution by Holiday Type', pad=20)
    
    # Monthly Holiday Pattern
    monthly_holiday = df[df['is_holiday']].groupby(df['Date'].dt.month).size()
    monthly_holiday.plot(kind='line', ax=ax4, color=NETFLIX_COLORS['primary'], 
                        marker='o')
    apply_netflix_style(ax4)
    ax4.set_title('Monthly Holiday Viewing Pattern', pad=20)
    
    plt.tight_layout()
    return holiday_views, top_holidays

# =============================================================================
# Analysis Runner
# =============================================================================
def run_all_analyses(netflix_data_cleaned):
    """Execute all analyses in sequence"""
    # Execute analyses
    trend_results = analyze_viewing_trends(netflix_data_cleaned)
    series_results = analyze_series_patterns(netflix_data_cleaned)
    content_results = analyze_content_distribution(netflix_data_cleaned)
    binge_results = analyze_binge_patterns(netflix_data_cleaned)
    seasonal_results = analyze_seasonal_patterns(netflix_data_cleaned)
    correlation_results = advanced_viewing_analysis(netflix_data_cleaned)
    advanced_trend_results = analyze_viewing_trends_advanced(netflix_data_cleaned)
    holiday_results = analyze_holiday_patterns(netflix_data_cleaned)
    trend_decomp_results = analyze_trends_and_seasonality(netflix_data_cleaned)
    
    return {
        'trend_results': trend_results,
        'series_results': series_results,
        'content_results': content_results,
        'binge_results': binge_results,
        'seasonal_results': seasonal_results,
        'correlation_results': correlation_results,
        'advanced_trend_results': advanced_trend_results,
        'holiday_results': holiday_results,
        'trend_decomp_results': trend_decomp_results
    }

def save_visualizations(results, netflix_data_cleaned):
    """Save all visualizations as separate PNG files"""
    viz_dir = '../output/visualizations'
    
    # Save visualization files
    plt.figure(figsize=(15, 15))
    trend_results = analyze_viewing_trends(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/viewing_trends.png')
    plt.close()
    
    # 2. Series Patterns
    plt.figure(figsize=(15, 10))
    series_results = analyze_series_patterns(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/series_patterns.png')
    plt.close()
    
    # 3. Content Distribution
    plt.figure(figsize=(15, 6))
    content_results = analyze_content_distribution(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/content_distribution.png')
    plt.close()
    
    # 4. Binge Patterns
    plt.figure(figsize=(15, 12))
    binge_results = analyze_binge_patterns(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/binge_patterns.png')
    plt.close()
    
    # 5. Seasonal Patterns
    plt.figure(figsize=(15, 6))
    seasonal_results = analyze_seasonal_patterns(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/seasonal_patterns.png')
    plt.close()
    
    # 6. Advanced Viewing Analysis
    plt.figure(figsize=(15, 12))
    correlation_results = advanced_viewing_analysis(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/advanced_viewing_analysis.png')
    plt.close()
    
    # 7. Advanced Trends
    plt.figure(figsize=(15, 10))
    trend_results = analyze_viewing_trends_advanced(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/advanced_trends.png')
    plt.close()
    
    # 8. Trends and Seasonality Analysis
    plt.figure(figsize=(15, 12))
    trend_results = analyze_trends_and_seasonality(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/trend_analysis.png')
    plt.close()
    
    # 9. Holiday Patterns
    plt.figure(figsize=(15, 12))
    holiday_results = analyze_holiday_patterns(netflix_data_cleaned)
    plt.savefig(f'{viz_dir}/holiday_patterns.png')
    plt.close()

# =============================================================================
# Main Analysis Execution
# =============================================================================
def main():
    """Main function to execute Netflix viewing analysis"""
    try:
        # 1. Load and preprocess data
        print("Loading Netflix viewing history data...")
        try:
            netflix_data = pd.read_csv('../data/NetflixViewingHistory.csv')
            print(f"Raw data columns: {', '.join(netflix_data.columns)}")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return None
            
        netflix_data_cleaned = preprocess_netflix_data(netflix_data)
        print(f"Loaded {len(netflix_data_cleaned)} viewing records\n")
        
        # 2. Run all analyses
        print("Running analyses...")
        results = run_all_analyses(netflix_data_cleaned)
        
        # 3. Save all visualizations
        print("\nSaving visualizations...")
        save_visualizations(results, netflix_data_cleaned)
        print("All visualizations have been saved as separate PNG files.")
        
        # 4. Print detailed insights
        print("\n" + "="*50)
        print("DETAILED NETFLIX VIEWING ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(f"Total viewing entries: {len(netflix_data_cleaned):,}")
        print(f"Date range: {netflix_data_cleaned['Date'].min()} to {netflix_data_cleaned['Date'].max()}")
        
        # Content distribution
        content_type = netflix_data_cleaned['Episode'].apply(
            lambda x: 'Movie' if x == 'Movie' else 'Series'
        ).value_counts()
        print("\nContent Distribution:")
        for content, count in content_type.items():
            print(f"{content}: {count:,} ({count/len(netflix_data_cleaned)*100:.1f}%)")
        
        # Series statistics
        series_counts = results['series_results'][0]
        print(f"\nUnique series watched: {len(series_counts):,}")
        print("\nTop 5 Most Watched Series:")
        for series, count in series_counts.head().items():
            print(f"- {series}: {count} episodes")
        
        # Binge watching statistics
        if results['binge_results'][1] is not None:
            binge_series = results['binge_results'][1]
            print("\nTop 3 Binge-Watched Series:")
            for series, count in binge_series.head(3).items():
                print(f"- {series}: {count} binge sessions")
        
        # Seasonal patterns
        seasonal_views = results['seasonal_results'][0]
        print(f"\nMost Active Viewing Season: {seasonal_views.idxmax()}")
        
        # Trend analysis
        trend_stats = results['advanced_trend_results']['trend_stats']
        print("\nViewing Trends:")
        print(f"Overall trend: {trend_stats['slope']:.2f} views per month")
        print(f"Trend significance (p-value): {trend_stats['p_value']:.4f}")
        print(f"Model fit (R-squared): {trend_stats['r_squared']:.3f}")
        
        # Add holiday analysis results to the output
        print("\nHoliday Viewing Patterns:")
        holiday_views = results['holiday_results'][0]
        print(f"Views on holidays: {holiday_views.get(True, 0):,}")
        print(f"Views on non-holidays: {holiday_views.get(False, 0):,}")
        
        top_holidays = results['holiday_results'][1]
        print("\nTop 3 Holiday Viewing Days:")
        for (date, holiday), views in top_holidays.head(3).items():
            print(f"- {holiday} ({date.strftime('%Y-%m-%d')}): {views} views")
        
        # Add trend decomposition results
        decomp, trend_stats = results['trend_decomp_results']
        print("\nTrend Decomposition Analysis:")
        print(f"Trend strength (R²): {trend_stats[2]**2:.3f}")
        print(f"Seasonal pattern detected: {'Yes' if decomp.seasonal.std() > 1 else 'No'}")
        
        print("\nAnalysis complete!")
        return results
        
    except FileNotFoundError:
        print("Error: Netflix viewing history file not found.")
        print("Please ensure 'NetflixViewingHistory.csv' is in the current directory.")
        return None
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    main()
