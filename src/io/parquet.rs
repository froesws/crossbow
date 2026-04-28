use std::fs::File;
use std::sync::Arc;
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::{Field, Schema};
use crate::{CrossbowError, DataFrame, Series};

/// Reads a Parquet file into a `DataFrame`.
///
/// All row groups are loaded and concatenated in memory.
pub fn read_parquet(path: &str) -> Result<DataFrame, CrossbowError> {
    let file = File::open(path).map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| CrossbowError::SchemaMismatch(e.to_string()))?;

    let schema = builder.schema().clone();
    let reader = builder
        .with_batch_size(65536)
        .build()
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

/// Writes a `DataFrame` to a Parquet file using Snappy compression.
pub fn write_parquet(df: &DataFrame, path: &str) -> Result<(), CrossbowError> {
    let schema = Arc::new(Schema::new(
        df.columns().iter().map(|s| {
            Field::new(s.name(), s.dtype().clone(), true)
        }).collect::<Vec<_>>()
    ));

    let arrays: Vec<Arc<dyn arrow::array::Array>> = df.columns()
        .iter()
        .map(|s| s.data().clone())
        .collect();

    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| CrossbowError::SchemaMismatch(e.to_string()))?;

    let file = File::create(path).map_err(|e| CrossbowError::IoError(e.to_string()))?;

    let props = parquet::file::properties::WriterProperties::builder()
        .set_compression(parquet::basic::Compression::SNAPPY)
        .build();

    let mut writer = parquet::arrow::ArrowWriter::try_new(
        file,
        schema,
        Some(props),
    ).map_err(|e| CrossbowError::IoError(e.to_string()))?;

    writer.write(&batch).map_err(|e| CrossbowError::IoError(e.to_string()))?;
    writer.close().map_err(|e| CrossbowError::IoError(e.to_string()))?;
    Ok(())
}
