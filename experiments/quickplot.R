#!/usr/bin/env Rscript
library(tidyverse)
library(ggdist)

db <- DBI::dbConnect(RSQLite::SQLite(), "~/kcmkc-results.sqlite")

tbl(db, "result") %>% 
  # filter(algorithm != "Random") %>%
  collect() %>% 
  mutate(tau = if_else(
    algorithm == "SeqCoreset", 
    as.integer(str_extract(algorithm_params, "\\d+")), 
    as.integer(0)
  )) %>% 
  ggplot(aes(factor(tau), radius, color=algorithm, fill=algorithm)) + 
  stat_summary(fun.data = mean_cl_boot, geom="linerange", size=1) +
  ggdist::stat_dots(binwidth=unit(0.01, "npc"), justification=-0.1) +
  ggdist::stat_dots(
    aes(y=coreset_radius), 
    binwidth=unit(0.01, "npc"), 
    side="left",
    justification=1.1,
    color="black",
    fill="black"
  ) +
  facet_wrap(vars(dataset), ncol=1, scales="free") +
  # scale_y_log10() +
  theme_bw()

ggsave("quick.png", width=9, height=4, dpi=90)
DBI::dbDisconnect(db)