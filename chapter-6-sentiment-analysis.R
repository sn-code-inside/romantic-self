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

# Visualise Moore and Galt with annotations
figure_6_4_data <- corpus %>% 
  bio_author_only() %>% 
  filter(biography %in% c("galt.xml", "moore.xml")) %>% 
  group_by(biography) %>% 
  mutate(
    score = zoo::rollmean(syuzhet, 250, fill = NA)
  )

# Indicate boundary between the two volumes of Moore's Byron
moore_volume_line <- corpus %>% 
  bio_author_only() %>% 
  filter(biography == "moore.xml") %>% 
  filter(str_detect(text, "The circumstances under which")) %>% 
  select(biography, sent_idx) %>% 
  mutate(
    label = c("Volume 1", "Volume 2") %>% list(),
    offset = c(sent_idx - 310, sent_idx + 310) %>% list(),
    y = -0.12
  )

# Annotate key plot points with text labels
create_label_row <-
  function(x,
           .new_data,
           .bio,
           .pattern,
           .off,
           .extract,
           .lab_off,
           .x_off = 0) {
    x %>%
      rows_append(
        .new_data %>%
          filter(biography == .bio) %>%
          filter(str_detect(text, .pattern)) %>%
          transmute(
            biography = biography,
            score = score,
            offset = score + .off,
            extract = .extract,
            label_offset = score + .lab_off,
            sent_idx = sent_idx,
            label_x = sent_idx + .x_off
          )
      )
  }

# Helper for finding local maxima on the graph
find_local_optimum <-
  function(data = figure_6_4_data,
           .biography = "moore.xml",
           .after = 0,
           .before = Inf,
           .optimum = max) {
    data %>%
      filter(biography == .biography,
             sent_idx >= .after,
             sent_idx <= .before,) %>%
      filter(score == .optimum(score, na.rm = TRUE))
  }

figure_6_4_annotations <- tibble::tibble(
  biography = as.character(),
  # for facet
  score = as.numeric(),
  # for 'y' parameter of geom_segment
  offset = as.numeric(),
  # for 'yend' parameter of geom_segment
  sent_idx = as.numeric(),
  # for x values of geom_segment
  extract = as.character(),
  # for 'label' of geom_text
  label_offset = as.numeric(),
  # for 'y' parameter of geom_text
  label_x = as.numeric()
  # for 'x' parameter of geom_text
) %>%
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "he sailed for Ostend",
    0.115,
    "'... he sailed\nfor Ostend.'",
    0.15
  ) %>%
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "he was no more!",
    0.13,
    "'... he was\nno more!'",
    0.16) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "bore him towards his beloved Greece",
    0.08,
    "'the breeze ... bore him\ntowards his beloved Greece'",
    0.11
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "galt.xml",
    "animal passions mastered",
    0.1,
    "His 'animal passions'\nmaster him",
    0.13
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "galt.xml",
    "the hollow valley",
    -0.09,
    "Byron gloomy and metaphysical\non his first trip to Greece",
    -0.125
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "galt.xml",
    "after committing murder",
    0.1,
    "Byron contemplates\nmurder",
    0.13
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "lone and unfriended",
    -0.1,
    "'lone and unfriended'\nin the House of Lords",
    -0.13
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "galt.xml",
    "chiefs of the factions",
    0.09,
    "Byron attempts to 'reconcile' the\n'factions' in Missolonghi",
    0.13
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "galt.xml",
    "never awoke again",
    0.1,
    "He dies",
    0.11
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "first had the happiness",
    0.1,
    "'It was at this period\nI first had the happiness\nof seeing ... Lord Byron'",
    0.15,
    200
  ) %>% 
  # create_label_row(
  #   figure_6_4_data,
  #   "moore.xml",
  #   "nights of the same description",
  #   0.1,
  #   "Moore hobnobs with Byron\nin London",
  #   0.13
  # ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "county of Durham",
    -0.12,
    "Byron marries\nAnnabella Milbanke",
    -0.145
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "she had breathed her last",
    -0.15,
    "Byron's mother dies",
    -0.16
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "eleven years from this period",
    0.16,
    "Byron swims the Hellespont",
    0.185
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "dictated by justice or by vanity",
    0.1,
    "The 'justice' of Byron's\nfeelings towards his wife",
    0.125
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "fair object of this last",
    -0.11,
    "Byron meets his 'last love',\n Teresa Guiccioli",
    -0.135
  ) %>% 
  # create_label_row(
  #   figure_6_4_data,
  #   "moore.xml",
  #   "in the shadow of the Alps",
  #   -0.1,
  #   "Byron defends himself in\n a bitter pamphlet",
  #   -0.125
  # ) %>% 
  # create_label_row(
  #   figure_6_4_data,
  #   "moore.xml",
  #   "all the followers of Pope",
  #   0.08,
  #   "Byron enthuses about\nAlexander Pope",
  #   0.105
  # ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "death of his daughter",
    -0.1,
    "Byron's daughter\nAllegra dies",
    -0.125
  ) %>% 
  create_label_row(
    figure_6_4_data,
    "moore.xml",
    "love of solitary rambles",
    0.1,
    "Young Byron's 'love of\nsolitary rambles'",
    0.15,
    15
  )

figure_6_4 <- figure_6_4_data %>% 
  ggplot(aes(sent_idx, score)) +
  geom_line() +
  geom_smooth(aes(sent_idx, syuzhet), method = "loess", alpha = 0.2) +
  facet_grid(rows = vars(biography), labeller = bio_labeller) +
  labs(
    x = "Sentence",
    y = "Sentiment score (syuzhet; rolling mean)"
  ) +
  geom_vline(
    aes(xintercept = sent_idx),
    linetype = 2,
    color = "darkgrey",
    data = moore_volume_line,
  ) +
  geom_text(
    aes(x = offset, label = label, y = y),
    data = unnest(moore_volume_line, cols = c(label, offset))
  ) +
  geom_segment(
    aes(x = sent_idx, xend = sent_idx, y = score, yend = offset),
    data = figure_6_4_annotations,
    color = "red"
  ) +
  geom_text(
    aes(x = label_x, y = label_offset, label = extract),
    data = figure_6_4_annotations
  ) 

ggsave("figures/figure_6_4.png", figure_6_4, width = 11, height = 7)
