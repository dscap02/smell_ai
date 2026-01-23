import argparse
import json
import os
import shutil
import subprocess
import sys
from matplotlib import pyplot as plt
import pandas as pd


class ReportGenerator:
    def __init__(self, input_path: str = ".", output_path: str = "."):
        """
        Initializes the ReportGenerator with input and output paths.

        Parameters:
        - input_path (str): The path to the
          directory or CSV file containing the input data.
        - output_path (str): The directory where the reports will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path

    def _find_project_details(self):
        """
        Checks if the input path is the 'project_details'
        folder or contains it.
        If found, gathers all CSV files for processing.

        Returns:
        - list: A list of CSV file paths within the
         'project_details' directory.
        """
        # Check if the input path is directly the `project_details` folder
        if os.path.basename(
            self.input_path
        ) == "project_details" and os.path.isdir(self.input_path):
            project_details_path = self.input_path
        else:
            # Search for a `project_details` folder inside the input path
            project_details_path = os.path.join(
                self.input_path, "project_details"
            )
            if not os.path.isdir(project_details_path):
                raise FileNotFoundError(
                    f"'project_details' folder not found in {self.input_path}."
                )

        # Gather all CSV files in the folder
        csv_files = [
            os.path.join(project_details_path, f)
            for f in os.listdir(project_details_path)
            if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {project_details_path}."
            )
        return csv_files

    def _load_data(self, file_paths):
        """
        Loads data from multiple CSV files into a single DataFrame.

        Parameters:
        - file_paths (list): List of file paths to load.

        Returns:
        - pd.DataFrame: A DataFrame containing all the merged data.
        """
        dfs = []
        for file in file_paths:
            print(f"Loading file: {file}")
            dfs.append(pd.read_csv(file))
        return pd.concat(dfs, ignore_index=True)

    def smell_report(self, df):
        """Generates a general overview report."""
        report = (
            df.groupby("smell_name")["filename"]
            .count()
            .rename("occurrences")
            .reset_index()
        )
        report.to_csv(
            os.path.join(self.output_path, "general_overview.csv"), index=False
        )
        print("General smell report saved to 'general_overview.csv'.")

    def project_report(self, df):
        """
        Generates a project-specific report
        treating files as part of separate projects.
        """
        # Extract project names from file paths
        df["project_name"] = df["filename"].apply(
            lambda x: os.path.basename(os.path.dirname(x)) or "root"
        )
        report = (
            df.groupby("project_name")["smell_name"]
            .count()
            .rename("total_smells")
            .reset_index()
        )
        output_file = os.path.join(self.output_path, "project_overview.csv")
        report.to_csv(output_file, index=False)
        print(f"Project-specific report saved to '{output_file}'.")

    def summary_report(self, df):
        """
        Generates a summary report with:
        - General overview of code smells.
        - Per-project summary of total smells.
        - Detailed sheets for each project.
        """
        df["project_name"] = df["filename"].apply(
            lambda x: os.path.basename(os.path.dirname(x)) or "root"
        )
        general_report = (
            df.groupby("smell_name")["filename"]
            .count()
            .rename("occurrences")
            .reset_index()
        )
        project_summary = (
            df.groupby("project_name")["smell_name"]
            .count()
            .rename("total_smells")
            .reset_index()
        )
        output_file = os.path.join(self.output_path, "summary_report.xlsx")
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            general_report.to_excel(
                writer, sheet_name="General Overview", index=False
            )
            project_summary.to_excel(
                writer, sheet_name="Project Overview", index=False
            )
            for project_name, project_df in df.groupby("project_name"):
                details = (
                    project_df.groupby("smell_name")["filename"]
                    .count()
                    .rename("occurrences")
                    .reset_index()
                )
                sanitized_name = project_name[
                    :30
                ]  # Excel sheet names must be <= 31 chars
                details.to_excel(
                    writer, sheet_name=sanitized_name, index=False
                )
        print(f"Summary report saved to '{output_file}'.")

    def visualize_smell_report(self, df):
        """Generates a bar chart for the general smell overview."""
        report = (
            df.groupby("smell_name")["filename"]
            .count()
            .rename("occurrences")
            .reset_index()
        )
        report.plot(kind="bar", x="smell_name", y="occurrences", legend=False)
        plt.title("Smell Occurrences by Type")
        plt.xlabel("Smell Type")
        plt.ylabel("Number of Occurrences")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "smell_report_chart.png"))
        print("Bar chart saved to 'smell_report_chart.png'.")

    def export_call_graph_metrics(self, analysis_results: dict) -> None:
        """
        Exports call graph metrics to a CSV file.
        """
        metrics = analysis_results.get("metrics", {})
        rows = []
        for node, values in metrics.items():
            rows.append(
                {
                    "node": node,
                    "in_degree": values.get("in_degree", 0),
                    "out_degree": values.get("out_degree", 0),
                    "betweenness_centrality": values.get(
                        "betweenness_centrality", 0.0
                    ),
                    "smell_count": values.get("smell_count", 0),
                }
            )
        df = pd.DataFrame(rows)
        output_file = os.path.join(
            self.output_path, "call_graph_metrics.csv"
        )
        df.to_csv(output_file, index=False)
        print(f"Call graph metrics saved to '{output_file}'.")

    def export_call_graph_cycles(self, analysis_results: dict) -> None:
        """
        Exports call graph cycles to a JSON file.
        """
        cycles = analysis_results.get("cycles", [])
        output_file = os.path.join(
            self.output_path, "call_graph_cycles.json"
        )
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(cycles, json_file, indent=2)
        print(f"Call graph cycles saved to '{output_file}'.")

    def visualize_call_graph(self, dot_path: str) -> None:
        """
        Generates SVG and PNG visualizations from a DOT file.
        """
        dot_binary = shutil.which("dot")
        if not dot_binary:
            print("Graphviz 'dot' binary not found. Skipping visualization.")
            return

        svg_path = os.path.join(self.output_path, "call_graph.svg")
        png_path = os.path.join(self.output_path, "call_graph.png")

        subprocess.run(
            [dot_binary, "-Tsvg", dot_path, "-o", svg_path], check=False
        )
        subprocess.run(
            [dot_binary, "-Tpng", dot_path, "-o", png_path], check=False
        )
        print(f"Call graph visualization saved to '{svg_path}'.")
        print(f"Call graph visualization saved to '{png_path}'.")

    def menu(self):
        """Displays a menu for the user to choose which report to generate."""
        print("Select which reports to generate:")
        print("1. General Smell Report")
        print("2. Project-Specific Report")
        print("3. Both Reports")
        print("4. Plot Report")
        print("5. Summary .xlsx Report")
        print("6. Exit")
        choice = input("Enter your choice: ")
        return choice

    def run(self):
        """
        Main execution logic for the report generation process.
        Handles user input and orchestrates report generation.
        """
        try:
            csv_files = self._find_project_details()
            df = self._load_data(csv_files)
            choice = self.menu()
            if choice == "1":
                self.smell_report(df)
            elif choice == "2":
                self.project_report(df)
            elif choice == "3":
                self.smell_report(df)
                self.project_report(df)
            elif choice == "4":
                self.visualize_smell_report(df)
            elif choice == "5":
                self.summary_report(df)
            elif choice == "6":
                print("Exiting...")
            else:
                print("Invalid choice. Exiting.")
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports for code smell analysis."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input directory or file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the reports.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a valid directory.")
        sys.exit(1)

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    generator = ReportGenerator(input_path=args.input, output_path=args.output)
    generator.run()


if __name__ == "__main__":
    main()
