do_latex_sample <- function(data) {
    data %>%
        filter(algorithm %in% c("SeqCoreset", "ChenEtAl")) %>%
        filter(str_detect(dataset, "sample")) %>%
        mutate(dataset = str_remove(dataset, "-sample-.*")) %>%
        group_by(dataset, rank, outliers_spec, 
                 algorithm, tau, algorithm_params) %>%
        summarise(
            ratio_to_best = mean(ratio_to_best),
            total_time = set_units(mean(total_time), "s") %>% drop_units()
        ) %>%
        ungroup() %>%
        group_by(dataset, algorithm) %>%
        slice_min(ratio_to_best) %>%
        slice_min(total_time) %>%
        ungroup() %>%
        select(dataset, algorithm, total_time, ratio_to_best) %>%
        pivot_wider(names_from=algorithm, values_from=c(total_time, ratio_to_best)) %>%
        kbl(format="latex", booktabs = T,
            escape = F,
            col.names = c("dataset", "\\chen", "\\seq", "\\chen", "\\chen")) %>%
        kable_styling() %>%
        add_header_above(c(" " = 1, "Total time (s)" = 2, "Ratio" = 2))
}
