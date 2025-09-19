mod crossbow;

use arrow::array::{Int32Array, StringArray};
use crossbow::Series;
use std::sync::Arc;

fn main() {
    // --- Exemplo de Criação de Series ---

    // 1. Criando uma Series de inteiros
    // let int_data = Int32Array::from(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
    let series_i32 = Series::from(vec![1, 2, 3, 4, 5], "numeros");
    //let series_i32 = Series::new("numeros", Arc::new(int_data));

    // 2. Criando uma Series de strings
    let str_data = StringArray::from(vec![Some("a"), Some("b"), Some("c"), Some("d"), None]);
    let series_str = Series::new("letras", Arc::new(str_data));

    println!("--- Informações da Series de Inteiros ---");
    println!("Nome: {}", series_i32.name());
    println!("Tipo de Dado: {:?}", series_i32.dtype());
    println!("Comprimento: {}", series_i32.len());

    println!("\n--- Informações da Series de Strings ---");
    println!("Nome: {}", series_str.name());
    println!("Tipo de Dado: {:?}", series_str.dtype());
    println!("Comprimento: {}", series_str.len());

    println!("\n--- Realizando uma Soma ---");
    if let Some(int_array) = series_i32.as_primitive::<Int32Array>() {
        let sum: Option<i32> = int_array.iter().sum();
        println!(
            "A soma da série '{}' é: {:?}",
            series_i32.name(),
            sum.unwrap_or(0)
        );
    } else {
        println!("A série '{}' não é do tipo Int32Array.", series_i32.name());
    }

    println!("\n--- Exibindo a Series de Inteiros ---");
    println!("{}", series_i32);

    println!("\n--- Exibindo a Series de Strings ---");
    println!("{}", series_str);
}
