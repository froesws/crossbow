use crossbow::DataFrame;

fn main() {
    println!("Executando o binário de demonstração do Crossbow!"); //

    let csv_path = "/home/froes/projetos/playground/employees.csv";
    println!("Tentando carregar '{}'...", csv_path);

    // Tenta carregar o CSV.
    // `read_csv` é a função que implementamos no DataFrame.
    match DataFrame::read_csv(csv_path) {
        Ok(df) => {
            println!("\n--- SUCESSO! ---");
            println!(
                "CSV carregado em memória. Usando Debug print (`{:#?}`):",
                df
            );

            // `df` aqui é a sua struct DataFrame.
            // O `Debug` print vai mostrar os nomes das colunas
            // e os enums `SeriesData` internos (ex: Int64(Vec<Option<i64>>)).
            println!("{:#?}", df); // O `#` no `{:#?}` faz um "pretty-print"

            // Agora podemos interagir com ele
            println!("\nShape do DataFrame: {:?}", df.shape());
            println!("Nomes das colunas: {:?}", df.get_column_names());

            // Vamos tentar um filtro
            println!("\nFiltrando por 'age > 25'...");
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
