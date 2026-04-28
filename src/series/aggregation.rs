use arrow::datatypes::DataType;
use arrow::array::Array;
use crate::error::CrossbowError;
use crate::series::Series;

impl Series {
    pub fn sum(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut sum = 0i64;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        sum += arr.value(i) as i64;
                    }
                }
                Ok(sum as f64)
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut sum_val = 0.0;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        sum_val += arr.value(i);
                    }
                }
                Ok(sum_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("sum() not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn mean(&self) -> Result<f64, CrossbowError> {
        match self.dtype() {
            DataType::Int32 | DataType::Float64 => {
                let sum = self.sum()?;
                let count = self.count_non_null()?;
                if count == 0 {
                    Ok(0.0)
                } else {
                    Ok(sum / count as f64)
                }
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("mean() not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn min(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut min_val: Option<i32> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        min_val = Some(match min_val {
                            None => val,
                            Some(m) => if val < m { val } else { m },
                        });
                    }
                }
                Ok(min_val.map(|v| v as f64))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut min_val: Option<f64> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        min_val = Some(match min_val {
                            None => val,
                            Some(m) => if val < m { val } else { m },
                        });
                    }
                }
                Ok(min_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("min() not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn max(&self) -> Result<Option<f64>, CrossbowError> {
        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                let mut max_val: Option<i32> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        max_val = Some(match max_val {
                            None => val,
                            Some(m) => if val > m { val } else { m },
                        });
                    }
                }
                Ok(max_val.map(|v| v as f64))
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                let mut max_val: Option<f64> = None;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        max_val = Some(match max_val {
                            None => val,
                            Some(m) => if val > m { val } else { m },
                        });
                    }
                }
                Ok(max_val)
            }
            _ => Err(CrossbowError::OperationNotSupported(
                format!("max() not supported for {:?}", self.dtype())
            )),
        }
    }

    pub fn count(&self) -> usize {
        self.len()
    }

    pub fn count_non_null(&self) -> Result<usize, CrossbowError> {
        let mut count = 0;
        for i in 0..self.len() {
            if self.data.is_valid(i) {
                count += 1;
            }
        }
        Ok(count)
    }

    pub fn std(&self) -> Result<f64, CrossbowError> {
        if self.count_non_null()? < 2 {
            return Ok(0.0);
        }

        let mean = self.mean()?;
        let mut variance = 0.0;
        let mut count = 0;

        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i) as f64;
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("std() not supported for {:?}", self.dtype())
                ));
            }
        }

        if count < 2 {
            return Ok(0.0);
        }

        Ok((variance / (count - 1) as f64).sqrt())
    }

    pub fn var(&self) -> Result<f64, CrossbowError> {
        if self.count_non_null()? < 2 {
            return Ok(0.0);
        }

        let mean = self.mean()?;
        let mut variance = 0.0;
        let mut count = 0;

        match self.dtype() {
            DataType::Int32 => {
                let arr = self.as_primitive::<arrow::array::Int32Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Int32Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i) as f64;
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            DataType::Float64 => {
                let arr = self.as_primitive::<arrow::array::Float64Array>().ok_or_else(
                    || CrossbowError::TypeMismatch("Expected Float64Array".to_string()),
                )?;
                for i in 0..arr.len() {
                    if arr.is_valid(i) {
                        let val = arr.value(i);
                        variance += (val - mean).powi(2);
                        count += 1;
                    }
                }
            }
            _ => {
                return Err(CrossbowError::OperationNotSupported(
                    format!("var() not supported for {:?}", self.dtype())
                ));
            }
        }

        if count < 2 {
            return Ok(0.0);
        }

        Ok(variance / (count - 1) as f64)
    }
}
