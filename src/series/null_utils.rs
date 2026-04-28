use std::sync::Arc;
use arrow::datatypes::DataType;
use arrow::array::Array;
use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    pub fn is_null(&self) -> Result<Series, CrossbowError> {
        let mut builder = arrow::array::BooleanBuilder::new();
        for i in 0..self.len() {
            builder.append_value(self.data.is_null(i));
        }
        Ok(Series::new(format!("{}_is_null", self.name()), Arc::new(builder.finish())))
    }

    pub fn is_not_null(&self) -> Result<Series, CrossbowError> {
        let mut builder = arrow::array::BooleanBuilder::new();
        for i in 0..self.len() {
            builder.append_value(!self.data.is_null(i));
        }
        Ok(Series::new(format!("{}_is_not_null", self.name()), Arc::new(builder.finish())))
    }

    pub fn fill_null(&self, fill_value: &Series) -> Result<Series, CrossbowError> {
        if self.len() != fill_value.len() {
            return Err(CrossbowError::MismatchedColumnLengths);
        }

        if self.dtype() != fill_value.dtype() {
            return Err(CrossbowError::TypeMismatch(
                format!("Cannot fill {:?} with {:?}", self.dtype(), fill_value.dtype())
            ));
        }

        match self.dtype() {
            DataType::Int32 => {
                let arr1 = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) {
                        builder.append_value(arr1.value(i));
                    } else if arr2.is_valid(i) {
                        builder.append_value(arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{}_filled", self.name()), Arc::new(builder.finish())))
            }
            DataType::Float64 => {
                let arr1 = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let arr2 = fill_value.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::new();
                for i in 0..arr1.len() {
                    if arr1.is_valid(i) {
                        builder.append_value(arr1.value(i));
                    } else if arr2.is_valid(i) {
                        builder.append_value(arr2.value(i));
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Series::new(format!("{}_filled", self.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("fill_null not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn drop_null(&self) -> Result<Series, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut builder = arrow::array::Int32Builder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut builder = arrow::array::Float64Builder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            DataType::Utf8 => {
                let arr = self.data.as_any().downcast_ref::<arrow::array::StringArray>()
                    .ok_or_else(|| CrossbowError::TypeMismatch("Expected StringArray".to_string()))?;
                let mut builder = arrow::array::StringBuilder::new();
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        builder.append_value(arr.value(i));
                    }
                }
                Ok(Series::new(format!("{}_no_null", self.name()), Arc::new(builder.finish())))
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("drop_null not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn value_at(&self, index: usize) -> Result<String, CrossbowError> {
        if index >= self.len() {
            return Err(CrossbowError::IndexOutOfBounds(index));
        }
        Ok(self.get_value_as_string(index))
    }

    pub fn slice(&self, offset: usize, length: usize) -> Result<Series, CrossbowError> {
        if offset + length > self.len() {
            return Err(CrossbowError::IndexOutOfBounds(offset + length));
        }
        let sliced = self.data.slice(offset, length);
        Ok(Series::new(format!("{}_slice", self.name()), Arc::new(sliced)))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self.dtype(), 
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
            DataType::Float32 | DataType::Float64
        )
    }

    pub fn is_string(&self) -> bool {
        matches!(self.dtype(), DataType::Utf8 | DataType::LargeUtf8)
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self.dtype(), DataType::Boolean)
    }
}
