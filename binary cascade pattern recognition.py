import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BinaryCascadeAnalyzer:
    def __init__(self, min_value=0, max_value=3):
        self.min_value = min_value
        self.max_value = max_value

    def generate_binary_string(self, length):
        return ''.join(str(random.randint(self.min_value, self.max_value)) for _ in range(length))

    def create_cascade(self, binary_string, window_size, start_position):
        cascade = []
        current_position = start_position
        while current_position < len(binary_string):
            row = binary_string[current_position:current_position + window_size]
            if len(row) == window_size:
                cascade.append(row)
            current_position += window_size
        return cascade

    def calculate_column_sums(self, cascade):
        if not cascade:
            return []
        return [sum(int(row[i]) for row in cascade) for i in range(len(cascade[0]))]

    def is_substantially_different(self, column_sums, base_threshold=1.5):
        if len(column_sums) < 2:
            return False
        mean = np.mean(column_sums)
        std = np.std(column_sums)
        if std == 0:
            return len(set(column_sums)) > 1  # Consider different if there are at least two distinct values
        z_scores = np.abs((np.array(column_sums) - mean) / std)
        adjusted_threshold = base_threshold * (1 - 0.1 * (self.max_value - self.min_value))  # Adjust threshold based on range
        return np.any(z_scores > adjusted_threshold)

    def run_experiment(self):
        binary_string = self.generate_binary_string(1000)
        max_window_size = min(self.max_value - self.min_value + 1, 50)
        for window_size in range(2, max_window_size + 1):
            for start_position in range(window_size):
                cascade = self.create_cascade(binary_string, window_size, start_position)
                if not cascade:
                    continue
                column_sums = self.calculate_column_sums(cascade)
                if self.is_substantially_different(column_sums):
                    return window_size
        return None  # Return None if no substantial difference is found

    def run_multiple_experiments(self, num_experiments):
        results = []
        for _ in range(num_experiments):
            result = self.run_experiment()
            if result is not None:
                results.append(result)
        return results

    def analyze_artificial_data(self, length=1000, true_pattern_size=7):
        pattern = [random.randint(self.min_value, self.max_value) for _ in range(true_pattern_size)]
        repeats = length // true_pattern_size + 1
        binary_string = ''.join(map(str, (pattern * repeats)[:length]))
        
        detected_sizes = []
        max_window_size = min(self.max_value - self.min_value + 1, 50)
        for window_size in range(2, max_window_size + 1):
            for start_position in range(window_size):
                cascade = self.create_cascade(binary_string, window_size, start_position)
                if not cascade:
                    continue
                column_sums = self.calculate_column_sums(cascade)
                if self.is_substantially_different(column_sums):
                    detected_sizes.append(window_size)
                    break
        return detected_sizes, true_pattern_size

def calculate_statistics(results):
    if not results:
        return {
            'average': None,
            'median': None,
            'std_dev': None,
            'substantially_different_ratio': 0
        }
    return {
        'average': np.mean(results),
        'median': np.median(results),
        'std_dev': np.std(results),
        'substantially_different_ratio': len(results) / 100
    }

def plot_histogram(results, min_value, max_value, title_suffix=""):
    if not results:
        print("No results to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(results, bins=range(min(results), max(results) + 2, 1), edgecolor='black')
    plt.title(f'Distribution of Window Sizes (Range {min_value}-{max_value}) {title_suffix}')
    plt.xlabel('Window Size')
    plt.ylabel('Frequency')
    plt.savefig(f'window_size_distribution_{min_value}_{max_value}{title_suffix.replace(" ", "_")}.png')
    plt.close()

def statistical_checks(results):
    if len(results) < 3:
        print("Not enough data for meaningful statistical checks.")
        return []

    checks = []
    try:
        checks.append(("Shapiro-Wilk Test for Normality", stats.shapiro(results)))
    except ValueError:
        checks.append(("Shapiro-Wilk Test for Normality", "Not applicable (insufficient data variation)"))

    try:
        checks.append(("Anderson-Darling Test for Normality", stats.anderson(results)))
    except ValueError:
        checks.append(("Anderson-Darling Test for Normality", "Not applicable (insufficient data variation)"))

    checks.extend([
        ("Skewness", stats.skew(results)),
        ("Kurtosis", stats.kurtosis(results)),
        ("Kolmogorov-Smirnov Test", stats.kstest(results, 'norm')),
    ])

    if len(results) >= 8:
        try:
            checks.append(("Chi-Square Test for Variance", stats.chisquare(results)))
        except ValueError:
            checks.append(("Chi-Square Test for Variance", "Not applicable (insufficient data variation)"))
        
        checks.extend([
            ("Jarque-Bera Test", stats.jarque_bera(results)),
            ("D'Agostino's K^2 Test", stats.normaltest(results)),
        ])

    if len(results) > 1:
        checks.append(("Mann-Whitney U Test", stats.mannwhitneyu(results[:len(results)//2], results[len(results)//2:])))

    checks.append(("Correlation Test", np.corrcoef(results, range(len(results)))[0, 1]))
    return checks

def main_random(min_value, max_value):
    random.seed(42)  # Set a fixed seed for reproducibility
    analyzer = BinaryCascadeAnalyzer(min_value, max_value)
    
    print(f"\nRunning analysis for range {min_value}-{max_value}")
    results = analyzer.run_multiple_experiments(100)
    
    if not results:
        print("No substantial differences detected in any experiment.")
        return
    
    stats = calculate_statistics(results)
    plot_histogram(results, min_value, max_value, " (Random Data)")
    
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nStatistical Checks:")
    checks = statistical_checks(results)
    for name, result in checks:  # Display all checks
        print(f"{name}: {result}")

def main_artificial_data(length=1000, true_pattern_size=7, min_value=0, max_value=9):
    random.seed(42)  # Set a fixed seed for reproducibility
    analyzer = BinaryCascadeAnalyzer(min_value, max_value)
    
    detected_sizes, true_size = analyzer.analyze_artificial_data(length, true_pattern_size)
    
    print(f"\nAnalysis of Artificial Data (True Pattern Size: {true_size})")
    print(f"Detected window sizes: {detected_sizes}")
    
    if true_size in detected_sizes:
        print(f"Success! True pattern size {true_size} was detected.")
    else:
        print(f"True pattern size {true_size} was not detected.")
    
    if not detected_sizes:
        print("No substantial differences detected.")
        return
    
    stats = calculate_statistics(detected_sizes)
    plot_histogram(detected_sizes, min_value, max_value, f" (Artificial Data, Pattern Size {true_size})")
    
    print("\nStatistics of Detected Sizes:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nStatistical Checks:")
    checks = statistical_checks(detected_sizes)
    for name, result in checks:  # Display all checks
        print(f"{name}: {result}")

if __name__ == "__main__":
    print("Analysis with Random Data:")
    main_random(0, 3)
    main_random(0, 9)
    
    print("\nAnalysis with Artificial Data:")
    main_artificial_data(length=1000, true_pattern_size=7, min_value=0, max_value=9)
    main_artificial_data(length=1000, true_pattern_size=4, min_value=0, max_value=3)