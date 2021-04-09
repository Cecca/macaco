plan <- drake_plan(
    data_result = {
        file_in("kcmkc-results.sqlite")
        table_result()
    },
    plot_tradeoff = target(
        do_plot_tradeoff(filter(data_result, rank == rank_value)),
        transform = cross(
            rank_value = c(10, 100)
        )
    ),
    plot_param_influence = do_plot_param_influence(data_result),
    notes = rmarkdown::render(
        knitr_in("R/notes.Rmd"),
        output_file = file_out("notes.html"),
        output_dir = "R",
        quiet = TRUE
    )
)