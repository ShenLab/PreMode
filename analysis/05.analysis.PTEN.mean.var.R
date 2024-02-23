pten.mean.var <- read.csv('PreMode.mean.var/PTEN/testing.fold.0.csv')
pten.raw.1 <- read.csv('/share/vault/Users/gz2294/Data/DMS/MAVEDB/PTEN/assay.1.csv', row.names = 1)
pten.raw.2 <- read.csv('/share/vault/Users/gz2294/Data/DMS/MAVEDB/PTEN/assay.2.csv', row.names = 1)

pten.mean.var$VarID <- paste0(pten.mean.var$ref, pten.mean.var$pos.orig, pten.mean.var$alt)
pten.raw.1$VarID <- paste0(pten.raw.1$ref, pten.raw.1$pos.orig, pten.raw.1$alt)
pten.raw.2$VarID <- paste0(pten.raw.2$ref, pten.raw.2$pos.orig, pten.raw.2$alt)

pten.mean.var$sd.1 <- pten.raw.1$sd[match(pten.mean.var$VarID, pten.raw.1$VarID)]
pten.mean.var$sd.2 <- pten.raw.2$sd[match(pten.mean.var$VarID, pten.raw.2$VarID)]

pten.mean.var$logits_var.0 <- log(1+exp(pten.mean.var$logits_var.0))
pten.mean.var$logits_var.1 <- log(1+exp(pten.mean.var$logits_var.1))

# first show average matched
sp.stat1 <- cor.test(pten.mean.var$logits.0, pten.mean.var$score.1, method = 'spearman')
p1 <- ggplot(pten.mean.var, aes(x=logits.0, y=score.1)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat1$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()
sp.stat2 <- cor.test(abs(pten.mean.var$logits.0-pten.mean.var$score.1),
                    pten.mean.var$logits_var.0, method = 'spearman')
p2 <- ggplot(pten.mean.var, aes(x=abs(logits.0-score.1), y=logits_var.0)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat2$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()

# first show average matched
sp.stat3 <- cor.test(pten.mean.var$logits.1, pten.mean.var$score.2, method = 'spearman')
p3 <- ggplot(pten.mean.var, aes(x=logits.1, y=score.2)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat3$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()
sp.stat4 <- cor.test(abs(pten.mean.var$logits.1-pten.mean.var$score.2),
                    pten.mean.var$logits_var.1, method = 'spearman')
p4 <- ggplot(pten.mean.var, aes(x=abs(logits.1-score.2), y=logits_var.1)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat4$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()
sp.stat5 <- cor.test(abs(pten.mean.var$logits.0-pten.mean.var$score.1),
                     pten.mean.var$score.1, method = 'spearman')
p5 <- ggplot(pten.mean.var, aes(x=abs(logits.0-score.1), y=score.1)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat5$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()
sp.stat6 <- cor.test(abs(pten.mean.var$logits.1-pten.mean.var$score.2),
                     pten.mean.var$score.2, method = 'spearman')
p6 <- ggplot(pten.mean.var, aes(x=abs(logits.1-score.2), y=score.2)) + 
  geom_point(alpha=0.5) + 
  geom_density_2d() + 
  stat_smooth(method = "lm", formula = y~x) +
  ggpubr::stat_regline_equation(
    aes(label =  paste0(paste(after_stat(eq.label), after_stat(adj.rr.label), sep = "~~~~"),
                        '~~~~', paste0("rho~`=`~", signif(sp.stat6$estimate, digits = 2)))),
    formula = y~x
  ) + theme_bw()
library(patchwork)
p <- p1 + p2 + p5 + p3 + p4 + p6 + plot_layout(ncol = 3)
ggsave('figs/05.PTEN.mean.var.pdf', p, height = 10, width = 15)

