/**
 * Facts quoted from the TRIBE v2 paper (d'Ascoli et al., FAIR at Meta, 25 March 2026). Figures are not reproduced;
 * the numbers are the paper's (Table 1 totals, section 2.3, section 5.8 and the discussion).
 */
export const TRIBE_PAPER = {
  title: 'A foundation model of vision, audition, and language for in-silico neuroscience',
  authors: "d'Ascoli et al., FAIR at Meta",
  date: '2026',
  facts: [
    { value: '720 people', text: '1,117 h of fMRI across 8 datasets (trained on 25 people and 452 h; the rest tests new people)' },
    { value: '3 inputs', text: 'video (V-JEPA2), audio (w2v-BERT) and text (Llama 3.2) features of the clip' },
    { value: 'R ≈ 0.4', text: 'on the 7T HCP set: about twice the median single person’s scan as a predictor of the group-average response' },
    { value: '5 s delay', text: 'the brain’s delay; our timelines shift predictions back so each point lines up with the moment that caused it' },
  ],
  limits: 'It models an average viewer’s perception at the time scale of fMRI (seconds). In the paper’s words it treats the brain as a passive observer, not an agent producing behaviour, so it does not model who keeps watching or shares.',
  links: [
    { label: 'Code', href: 'https://github.com/facebookresearch/tribev2' },
    { label: 'Weights', href: 'https://huggingface.co/facebook/tribev2' },
    { label: 'Meta demo', href: 'https://aidemos.atmeta.com/tribev2' },
  ],
} as const;
