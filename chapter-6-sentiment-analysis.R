# Sentiment analysis of the biographies in the corpus using syuzhet and vader
USE_SAVED_SCORES <- TRUE
library(tidyverse)

get_vader_sentiment <- function(text) {
  purrr::map_chr(
    text,
    \(x) vader::get_vader(x)[2], # compound score at idx 2,
    .progress = "Vader scores"
  )
}

# Based on the normalisation function from the vader package
# https://github.com/csrajath/vaderSentiment/blob/master/vaderSentiment/vaderSentiment.py#L106
normalise <- function(x, alpha = 15) {
  norm_score <- x / sqrt((x * x) + alpha)
  norm_score <- ifelse(norm_score > 1.0, 1.0, norm_score)
  norm_score <- ifelse(norm_score < -1.0, -1.0, norm_score)
  norm_score
}

bio_author_only <- function(corpus) {
  corpus %>%
    filter(sent_author == bio_author) %>% 
    group_by(biography) %>%
    mutate(sent_idx = seq(n())) %>% 
    ungroup()
}

if (!USE_SAVED_SCORES) {
  corpus <- readr::read_csv("data/biography-corpus/all-sentences.csv") %>% 
    mutate(
      syuzhet = syuzhet::get_sentiment(text, method="syuzhet"),
      bing = syuzhet::get_sentiment(text, method="bing"),
      afinn = syuzhet::get_sentiment(text, method="afinn"),
      vader = get_vader_sentiment(text)
    ) %>% 
    mutate(
      across(syuzhet:afinn, normalise),
      vader = as.numeric(vader)
    )
  
  write_csv(corpus, "data/biography-corpus/scored-sentences.csv")
} else {
  corpus <- readr::read_csv("data/biography-corpus/scored-sentences.csv")
}

# Figure 6.2: Plot all the biographies
bio_labeller <- as_labeller(function(x) str_to_title(x) %>% str_remove("\\.xml|\\.txt"))

break_every <- function(n) {
  function(limits) {
    lower = limits[1]
    upper = limits[2]
    num_breaks = upper %/% n
    rep
  }
}

figure_6_2 <- corpus %>% 
  bio_author_only() %>% 
  pivot_longer(syuzhet:vader, names_to = "model", values_to = "score") %>% 
  group_by(biography, model) %>%
  mutate(score = zoo::rollmean(score, k = 250, fill = NA)) %>%
  ungroup() %>% 
  ggplot(aes(sent_idx, score)) +
  facet_grid(vars(model), vars(biography), scales = "free_x", space = "free_x", labeller = bio_labeller) +
  scale_x_continuous(breaks = scales::breaks_width(2000)) +
  geom_line() +
  labs(
    x = "Sentence",
    y = "Sentiment score (rolling mean)"
  )

ggsave("figures/figure_6_2.png", plot = figure_6_2, width = 13, height = 9)

# Figure 6.3: Plot of Moore's biography in detail
corpus %>% 
  filter(biography == "moore.xml") %>% 
  mutate(across(syuzhet:vader, \(x) zoo::rollmean(x, k = 500, fill = NA))) %>% 
  pivot_longer(syuzhet:vader, names_to = "model", values_to = "score") %>% 
  mutate(
    sent_author = case_when(
      sent_author == "ThMoore1852" ~ "Moore",
      sent_author == "LdByron" ~ "Byron",
      .default = "Other"
    )
  ) %>% 
  ggplot(aes(sent_idx, score, color = sent_author)) +
  scale_color_grey() +
  facet_wrap(vars(model)) +
  geom_path(aes(group = 1))

# Are the sentiment scores normally distributed?
corpus %>% 
  bio_author_only() %>% 
  pivot_longer(syuzhet:vader, names_to = "model", values_to = "score") %>% 
  ggplot(aes(score, after_stat(density))) +
  facet_grid(vars(biography), vars(model)) +
  geom_histogram(bins = 10)

# Figure 6.3: Tabulate some statistics
figure_6_3 <- corpus %>% 
  bio_author_only() %>% 
  pivot_longer(syuzhet:vader, names_to = "model", values_to = "score") %>% 
  group_by(biography, model) %>% 
  summarise(
    mean = mean(score, na.rm = T),
    sd = sd(score, na.rm = T),
    range = diff(range(score, na.rm = T)),
    gradient = coef(lm(score ~ sent_idx))[2] * 10000, # slope per 10000 sentences
    .groups = "drop"
  ) %>% 
  group_by(model, .drop = T) %>% 
  mutate(
    across(mean:gradient, rank, .names="{.col}_rank")
  )

# Only export the mean, sd and range
figure_6_3 %>% 
  select(biography:range) %>% 
  mutate(across(where(is.numeric), \(x) round(x, digits = 3))) %>% 
  write_csv("figures/figure_6_3.csv")

# Difference between Moore and Galt
figure_6_4 %>% 
  filter(biography %in% c("moore.xml", "galt.xml")) %>% 
  group_by(model) %>% 
  summarise(
    mean = diff(mean),
    sd = diff(sd)
  )

# Visualise Moore and Galt with annotations
figure_6_4_data <- corpus %>% 
  bio_author_only() %>% 
  filter(biography %in% c("galt.xml", "moore.xml")) %>% 
  group_by(biography) %>% 
  mutate(
    score = zoo::rollmean(syuzhet, 250, fill = NA)
  )

figure_6_4_data %>% 
  ggplot(aes(sent_idx, score)) +
  geom_line() +
  geom_smooth(aes(sent_idx, syuzhet), method = "gam", alpha = 0.2) +
  facet_grid(rows = vars(biography), labeller = bio_labeller) +
  labs(
    x = "Sentence",
    y = "Sentiment score (syuzhet; rolling mean)"
  )
