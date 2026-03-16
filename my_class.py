from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.types import *
import pandas as pd
from pandas.api.types import is_numeric_dtype
from functools import reduce

class SparkDataCheck:
    """
    A data quality class for Spark SQL style data frames
    """
    #initializaiton and crating the df attribute
    def __init__(self, df: DataFrame):
        self.df = df
        
    def __getattr__(self, name):
        #Called when attribute 'name' isn't found on SparkDataCheck
        return getattr(self.df, name)
    
    @classmethod
    #creates an instance while reading in a csv file
    def from_spark(cls, spark, file_path: str, format: str, sep, header: bool = True, inferSchema: bool = True):
        df = spark.read.load(file_path, format=format, header=header, inferSchema=inferSchema, sep=sep)
        return cls(df)

    #create methods for creating a new instance of the class while reading in data
    @classmethod
    def from_pandas(cls, spark, pandas_df):
        df = spark.createDataFrame(pandas_df)
        return cls(df)

##########################################
    #validation methods
    #create boolean column based on numeric bounds
    def check_numeric_range(self, column: str, lower: float = None,  upper: float = None, inclusive: str = "both", new_column: str = None) -> DataFrame:
        """
        Append Boolean column indicating whether numeric values fall within bounds.
        For any NULL values in the checked column, the corresponding value in the new column will be NULL.
        """
        # Check if the column exists
        try:
            col_schema = self.df.schema[column]
        except KeyError:
            print(f"Error: Column '{column}' not found in DataFrame.")
            return self.df

        # Check if the column is numeric
        numeric_types = (DoubleType, IntegerType, LongType, FloatType)
        if not isinstance(col_schema.dataType, numeric_types):
            print(f"Warning: Column '{column}' is not a numeric type ({col_schema.dataType}). Returning DataFrame without modification.")
            return self.df

        # Check that at least one bound is provided
        if lower is None and upper is None:
            print("Error: At least one of 'lower' or 'upper' must be provided.")
            return self.df

        column_to_check = F.col(column)  #dynamic column reference
        condition = F.lit(True)          #Initialize a boolean column that is always True

        if lower is not None:
            if inclusive == "both" or inclusive == "left":
                condition = condition & (column_to_check >= lower)
            elif inclusive == "right" or inclusive == "neither":
                condition = condition & (column_to_check > lower)

        if upper is not None:
            if inclusive == "both" or inclusive == "right":
                condition = condition & (column_to_check <= upper)
            elif inclusive == "left" or inclusive == "neither":
                condition = condition & (column_to_check < upper)

        # Generate default new column name if not provided
        if new_column is None:
            parts = [column, "in_range"]
            parts.extend([str(val) for val in [lower, upper] if val is not None])
            new_column = "_".join(parts).replace('.', '_point_') # Handle floats in name

        return self.df.withColumn(new_column, condition)

    #create boolean column to check string levels
    def check_string_levels(self, column: str, levels: list, new_column: str = None) -> DataFrame:
        """
        Append a nullable-boolean column indicating whether each non-null value in the
        string `column` is in `levels`. For nulls, return null in the flag.
        """
        # Attempt to get the schema for the specified column.
        # If the column does not exist, a KeyError is caught.
        try:
            col_schema = self.df.schema[column]
        except KeyError:
            print(f"Error: Column '{column}' not found in DataFrame.")
            return self.df    # Return the original df without modification

        # Check if the column's data type is StringType.
        # If it's not a string, print a warning and return the original DataFrame.
        if not isinstance(col_schema.dataType, StringType):
            print(f"Warning: Column '{column}' is not a string type ({col_schema.dataType}). Returning DataFrame without modification.")
            return self.df

        # Create a boolean condition: True if the column's value is in the 'levels' list, False otherwise.
        # PySpark's .isin() method handles NULLs by returning NULL for the condition.
        condition = F.col(column).isin(levels)

        # If no new column name is provided, generate a default name.
        if new_column is None:
            new_column = f"{column}_in_levels"

        # Add the new boolean column to the DataFrame and return the modified DataFrame.
        return self.df.withColumn(new_column, condition)

    #create a method to check missing values
    def check_missing(self, column: str, new_column: str = None):
        """
        Create a method that checks if a each value in a column is missing (NULL specifically) and returns
        the dataframe with an appended column of Boolean values.
        """
        condition = F.col(column).isNull()

        # If no new column name is provided, generate a default name.
        if new_column is None:
            new_column = f"{column}_is_missing"

        return self.df.withColumn(new_column, condition)

    ############################
    #create a couple of summarization methods
    def count_min_max(self, column: str = None, group_by_col: str = None) -> pd.DataFrame:
        """
        Reports min and max of a numeric column (or all numeric columns) as a pandas DataFrame.
        - If `column` is provided:
            * must exist and be numeric; else print and return None
            * compute min/max (grouped if group_by_col)
        - If `column` is None:
            * compute min/max for ALL numeric columns (grouped if group_by_col)
            * produce NO messages; if no numeric cols, return empty pandas DataFrame
        """
        # Validate grouping column if provided
        if group_by_col:
            if group_by_col not in self.df.columns:
                print(f"Error: Grouping column '{group_by_col}' not found in DataFrame.")
                return None

        numeric_types = (DoubleType, IntegerType, LongType, FloatType)

        # Single user-specified column
        if column:
            #check if the column exists
            if column not in self.df.columns:
                print(f"Error: Column '{column}' not found in DataFrame.")
                return None
            #check if the column is numeric
            col_dtype = self.df.schema[column].dataType
            if not isinstance(col_dtype, numeric_types):
                print(f"Warning: Column '{column}' is not a numeric type ({col_dtype}). Returning None.")
                return None

            # Build one Spark DF with aliases first, then toPandas
            if group_by_col:  #grouped if appropriate
                spark_out = self.df.groupBy(group_by_col).agg(
                    F.min(column).alias(f"min_{column}"),
                    F.max(column).alias(f"max_{column}")
                )
            else:             #report min and max without grouping
                spark_out = self.df.agg(
                    F.min(column).alias(f"min_{column}"),
                    F.max(column).alias(f"max_{column}")
                )
            # to cover both grouped and ungrouped cases for a single column
            return spark_out.toPandas()

        # No column supplied → all numeric columns
        # This block executes if 'column' was None.
        num_cols = [f.name for f in self.df.schema.fields if isinstance(f.dataType, numeric_types)]
        if not num_cols:
            return pd.DataFrame() # if no numeric cols, return empty pandas DataFrame

        if not group_by_col:
            # Ungrouped: aggregate all at once and return to pandas
            spark_out = self.df.agg(
                *[F.min(c).alias(f"min_{c}") for c in num_cols],
                *[F.max(c).alias(f"max_{c}") for c in num_cols]
            )
            return spark_out.toPandas()

        # Grouped + all numeric: build per-column grouped results, then merge in pandas
        per_col_pandas = []
        for c in num_cols:
            sdf = self.df.groupBy(group_by_col).agg(
                F.min(c).alias(f"min_{c}"),
                F.max(c).alias(f"max_{c}"))
            pdf = sdf.toPandas()
            per_col_pandas.append(pdf)

        # use reduce() and pd.merge() to put the results into a single data frame
        merged_pdf = reduce(lambda left, right: pd.merge(left, right, on=group_by_col, how="outer"),
            per_col_pandas
        )
        # sort by group for readability
        merged_pdf = merged_pdf.sort_values(by=[group_by_col]).reset_index(drop=True)
        return merged_pdf
    #create a method to report the counts with string columns
    def counts_string(self, col1: str, col2:str = None) -> DataFrame:
        # List to store columns to group by
        cols_to_group = [col1]
        if col2 is not None:
            cols_to_group.append(col2)

        # Check if all specified columns exist and are of StringType
        for col_name in cols_to_group:
            # Check if the column exists
            try:
                col_schema = self.df.schema[col_name]
            except KeyError:
                print(f"Error: Column '{col_name}' not found in DataFrame.")
                return None

            # Check if the column is a StringType
            if not isinstance(col_schema.dataType, StringType):
                print(f"Warning: Column '{col_name}' is not a string type ({col_schema.dataType}). Returning None.")
                return None

        # If all checks pass, perform the grouping and counting
        count_df = self.df.groupBy(*cols_to_group).count()
        return count_df
          