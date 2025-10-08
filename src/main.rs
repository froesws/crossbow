use crossbow::{dataframe::DataFrame, series::Series};

fn main() {
    // Aqui você pode colar o código que estava na sua função main original.
    println!("Executando o binário de demonstração do Crossbow!");

    let s1 = Series::from("col A", vec![1, 2, 3]);
    let s2 = Series::from("col B", vec!["a", "b", "c"]);

    match DataFrame::new(vec![s1, s2]) {
        Ok(df) => {
            println!("DataFrame criado com sucesso a partir da biblioteca!");
            println!("{:?}", df);
        }
        Err(e) => {
            println!("Erro: {}", e);
        }
    }
}
