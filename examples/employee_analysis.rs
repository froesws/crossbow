use std::sync::Arc;
use std::fs::File;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::*;
use arrow::array::*;
use crossbow::{DataFrame, Series};

fn load_employee_data() -> DataFrame {
    let schema = Arc::new(Schema::new(vec![
        Field::new("employee", DataType::Utf8, true),
        Field::new("salary", DataType::Float64, true),
        Field::new("age", DataType::Int32, true),
        Field::new("position", DataType::Utf8, true),
        Field::new("yield", DataType::Int32, true),
    ]));

    let file = File::open("data_1M.csv").unwrap();
    let mut reader = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(100_000)
        .build(file)
        .unwrap();

    let batch = reader.next().unwrap().unwrap();

    let columns: Vec<Series> = schema.fields().iter().enumerate().map(|(i, field)| {
        Series::new(field.name(), batch.column(i).clone())
    }).collect();

    DataFrame::new(columns).unwrap()
}

fn make_int_mask(series: &Series, pred: fn(i32) -> bool) -> Series {
    let arr = series.as_primitive::<Int32Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn make_float_mask(series: &Series, pred: fn(f64) -> bool) -> Series {
    let arr = series.as_primitive::<Float64Array>().unwrap();
    let mut b = BooleanArray::builder(arr.len());
    for i in 0..arr.len() {
        b.append_value(arr.is_valid(i) && pred(arr.value(i)));
    }
    Series::new("mask", Arc::new(b.finish()))
}

fn print_separator(title: &str) {
    println!("\n{}", "=".repeat(70));
    println!("  {}", title);
    println!("{}", "=".repeat(70));
}

fn main() {
    println!("Loading employee dataset...");
    let df = load_employee_data();
    println!("Loaded {} employees with {} columns", df.shape().0, df.shape().1);

    println!("\nFirst look at the data:");
    println!("{}", df);

    // ---- Basic Column Analysis ----
    print_separator("Salary Statistics");
    let salary = df.select("salary").unwrap();
    println!("Mean salary:   ${:>9.2}", salary.mean().unwrap());
    println!("Min salary:    ${:>9.2}", salary.min().unwrap().unwrap_or(0.0));
    println!("Max salary:    ${:>9.2}", salary.max().unwrap().unwrap_or(0.0));
    println!("Std deviation: ${:>9.2}", salary.std().unwrap());
    println!("Variance:      ${:>9.2}", salary.var().unwrap());

    print_separator("Age Statistics");
    let age = df.select("age").unwrap();
    println!("Mean age:     {:>6.1} years", age.mean().unwrap());
    println!("Min age:      {:>6.0} years", age.min().unwrap().unwrap_or(0.0));
    println!("Max age:      {:>6.0} years", age.max().unwrap().unwrap_or(0.0));
    println!("Std deviation:{:>6.1} years", age.std().unwrap());

    print_separator("Performance (Yield) Statistics");
    let yield_col = df.select("yield").unwrap();
    println!("Mean yield:   {:>8.1}", yield_col.mean().unwrap());
    println!("Min yield:    {:>8.0}", yield_col.min().unwrap().unwrap_or(0.0));
    println!("Max yield:    {:>8.0}", yield_col.max().unwrap().unwrap_or(0.0));

    println!("\nTotal rows checked with count(): {}", age.count());
    println!("Non-null rows: {}", age.count_non_null().unwrap());

    // ---- Position Analysis ----
    print_separator("Employee Count by Position");
    let by_position = df.group_by("position").unwrap();
    let position_counts = by_position.count().unwrap();
    println!("{}", position_counts);
    println!("(Showing {} positions with their employee counts)", position_counts.shape().0);

    print_separator("Average Salary by Position");
    let salary_by_position = by_position.mean("salary").unwrap();
    println!("{}", salary_by_position);

    print_separator("Total Yield by Position");
    let yield_by_position = by_position.sum("yield").unwrap();
    println!("{}", yield_by_position);

    // ---- Filtering ----
    print_separator("Senior Employees (age >= 50)");
    let age_mask = make_int_mask(age, |v| v >= 50);
    let senior = df.filter_by_mask(&age_mask).unwrap();
    println!("Found {} senior employees", senior.shape().0);
    let senior_salary = senior.select("salary").unwrap();
    println!("Senior mean salary: ${:>9.2}", senior_salary.mean().unwrap());
    println!("Senior max salary:  ${:>9.2}", senior_salary.max().unwrap().unwrap_or(0.0));

    print_separator("High Performers (yield > 9000)");
    let high_yield_mask = make_int_mask(yield_col, |v| v > 9000);
    let high_performers = df.filter_by_mask(&high_yield_mask).unwrap();
    println!("Found {} high performers", high_performers.shape().0);
    let hp_yield = high_performers.select("yield").unwrap();
    println!("Mean yield: {}", hp_yield.mean().unwrap());

    // ---- Combined Filtering ----
    print_separator("Combined: age > 35 AND salary > $20,000");
    let age_gt_35 = make_int_mask(df.select("age").unwrap(), |v| v > 35);
    let step1 = df.filter_by_mask(&age_gt_35).unwrap();
    let salary_gt_20k = make_float_mask(step1.select("salary").unwrap(), |v| v > 20000.0);
    let step2 = step1.filter_by_mask(&salary_gt_20k).unwrap();
    println!("Found {} employees", step2.shape().0);
    for i in 0..step2.shape().0.min(5) {
        let r = step2.get_row(i).unwrap();
        println!("  {} | age {} | salary {} | position {}",
            r[0], r[2], r[1], r[3]);
    }

    // ---- Sorting ----
    print_separator("Top 10 Highest Paid Employees");
    let sorted = df.sort_by_desc("salary").unwrap();
    let top10 = sorted.get_rows(&(0..10).collect::<Vec<_>>()).unwrap();
    println!("{:<5} {:<40} {:>10} {:>5} {:>10}", "#", "Employee", "Salary", "Age", "Yield");
    for (i, row) in top10.iter().enumerate() {
        println!("{:<5} {:<40} {:>10} {:>5} {:>10}", i + 1, row[0], row[1], row[2], row[4]);
    }

    print_separator("Youngest Employees (sorted by age)");
    let youngest = df.sort_by_asc("age").unwrap()
        .get_rows(&(0..5).collect::<Vec<_>>()).unwrap();
    for (i, row) in youngest.iter().enumerate() {
        println!("#{}: {} (age {}, {}, yield: {})",
            i + 1, row[0], row[2], row[3], row[4]);
    }

    // ---- Arithmetic ----
    print_separator("Salary Projection (current + $5000)");
    let bonus = Series::from("bonus", vec![5000.0f64; df.shape().0]);
    let projected = salary.add(&bonus).unwrap();
    println!("Original mean salary:  ${:>9.2}", salary.mean().unwrap());
    println!("Projected mean salary: ${:>9.2}", projected.mean().unwrap());
    println!("Difference:            ${:>9.2}", projected.mean().unwrap() - salary.mean().unwrap());

    // ---- Arithmetic: Halve the yield ----
    print_separator("Yield Projection (dividing by 2)");
    let two = Series::from("two", vec![2i32; df.shape().0]);
    let half_yield = yield_col.divide(&two).unwrap();
    println!("Original mean yield: {:>8.1}", yield_col.mean().unwrap());
    println!("Halved mean yield:   {:>8.1}", half_yield.mean().unwrap());

    // ---- Column Management ----
    print_separator("Enhanced DataFrame (adding/removing columns)");
    let bonus_int = Series::from("bonus", vec![5000i32; df.shape().0]);
    let enhanced = df.add_column(bonus_int).unwrap();
    println!("Columns after add: {:?}", enhanced.get_column_names());

    let renamed = enhanced.rename_column("yield", "performance").unwrap();
    println!("Columns after rename: {:?}", renamed.get_column_names());

    let cleaned = renamed.remove_column("bonus").unwrap();
    println!("After remove 'bonus': {:?}", cleaned.get_column_names());

    // ---- Series value access ----
    print_separator("Individual Value Access");
    println!("First employee:  {}", df.select("employee").unwrap().value_at(0).unwrap());
    println!("Employee at 50:  {}", df.select("employee").unwrap().value_at(50).unwrap());

    // ---- Slice ----
    let first_100_ages = df.select("age").unwrap().slice(0, 100).unwrap();
    println!("\nMean age of first 100 employees: {:.1}", first_100_ages.mean().unwrap());
    println!("Min age in slice: {}", first_100_ages.min().unwrap().unwrap_or(0.0));
    println!("Max age in slice: {}", first_100_ages.max().unwrap().unwrap_or(0.0));

    // ---- Boolean mask: all true / all false ----
    print_separator("Edge Cases");
    let all_true = Series::new("all", Arc::new(BooleanArray::from(vec![true; df.shape().0])));
    let all = df.filter_by_mask(&all_true).unwrap();
    println!("Filter with all-true mask: {} rows (same as original)", all.shape().0);

    let all_false = Series::new("none", Arc::new(BooleanArray::from(vec![false; df.shape().0])));
    let none = df.filter_by_mask(&all_false).unwrap();
    println!("Filter with all-false mask: {} rows", none.shape().0);

    println!("\nAnalysis complete!");
}
