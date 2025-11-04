use crossbow::DataFrame;

fn main() {
    println!("Executando o binário de demonstração do Crossbow!"); //

    let csv_path = "employees.csv";
    println!("Tentando carregar '{}'...", csv_path);

    match DataFrame::read_csv(csv_path) {
        Ok(df) => {
            println!("\n--- SUCESSO! ---");
            println!(
                "CSV carregado em memória. Usando Debug print (`{:#?}`):",
                df
            );

            println!("\nShape do DataFrame: {:?}", df.shape());
            println!("Nomes das colunas: {:?}", df.get_column_names());

            if let (Ok(_age_col), Ok(mask)) = (
                df.select("age"),
                df.select("age").and_then(|col| col.gt_i64(25)),
            ) {
                println!("Máscara de filtro (Debug): {:?}", mask);
                if let Ok(filtered_df) = df.filter(&mask) {
                    println!("DataFrame Filtrado (Debug):");
                    println!("{:#?}", filtered_df);
                }
            }
        }
        Err(e) => {
            println!("\n--- ERRO! ---");
            println!("Não foi possível ler o CSV: {}", e);
            println!(
                "(Verifique se o arquivo '{}' existe na raiz do projeto)",
                csv_path
            );
        }
    }
}
