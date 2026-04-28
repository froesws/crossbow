use std::fs::File;
use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatchWriter;
use crate::{CrossbowError, DataFrame, Series};

fn build_schema(df: &DataFrame) -> Arc<Schema> {
    let fields: Vec<Field> = df.columns()
        .iter()
        .map(|s| Field::new(s.name(), s.dtype().clone(), true))
        .collect();
    Arc::new(Schema::new(fields))
}

fn df_to_record_batch(df: &DataFrame) -> Result<RecordBatch, CrossbowError> {
    let schema = build_schema(df);
    let arrays: Vec<Arc<dyn arrow::array::Array>> = df.columns()
        .iter()
        .map(|s| s.data().clone())
        .collect();
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| CrossbowError::SchemaMismatch(e.to_string()))
}

pub fn read_csv(path: &str) -> Result<DataFrame, CrossbowError> {
    let file = File::open(path).map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let (schema, _records) = arrow::csv::reader::Format::default()
        .with_header(true)
        .infer_schema(file, None)
        .map_err(|e| CrossbowError::SchemaMismatch(e.to_string()))?;

    let schema = Arc::new(schema);
    let file = File::open(path).map_err(|e| CrossbowError::IoError(e.to_string()))?;
    let reader = ReaderBuilder::new(schema.clone())
        .with_header(true)
        .with_batch_size(65536)
        .build(file)
        .map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let batches: Vec<RecordBatch> = reader
        .collect::<Result<_, _>>()
        .map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let batch = concat_batches(&schema, batches.iter())
        .map_err(|e| CrossbowError::SchemaMismatch(e.to_string()))?;

    let columns: Vec<Series> = schema.fields().iter().enumerate().map(|(i, field)| {
        Series::new(field.name(), batch.column(i).clone())
    }).collect();

    DataFrame::new(columns)
}

pub fn write_csv(df: &DataFrame, path: &str) -> Result<(), CrossbowError> {
    let batch = df_to_record_batch(df)?;
    let file = File::create(path).map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let mut writer = arrow::csv::WriterBuilder::new()
        .with_header(true)
        .build(file);

    writer.write(&batch).map_err(|e| CrossbowError::IoError(e.to_string()))?;
    writer.close().map_err(|e| CrossbowError::IoError(e.to_string()))?;
    Ok(())
}
