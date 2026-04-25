# Smadex Creative Intelligence Challenge Dataset

This is a **fully synthetic** ad-tech dataset created for a hackathon-style "Creative Intelligence" challenge.
It is intentionally **large enough to feel real**, but still **bounded enough for university teams** to understand in a few minutes.

## What teams can build
- A dashboard to explain why some creatives work better than others
- A fatigue detector that spots when a creative is wearing out
- A recommendation engine for the next creative variation to test
- A lightweight scoring model for creative quality by vertical, country or OS

## Recommended framing
Do **not** treat this like a perfect prediction competition.
The most interesting projects will combine:
1. analysis,
2. simple modeling,
3. explainability,
4. a product demo.

## Files
- `advertisers.csv`: advertiser-level metadata
- `campaigns.csv`: campaign setup and targeting
- `creatives.csv`: creative metadata plus the relative path of each synthetic PNG asset
- `creative_daily_country_os_stats.csv`: main fact table with one row per date x creative x country x OS
- `creative_summary.csv`: pre-aggregated creative-level metrics and a synthetic status label
- `campaign_summary.csv`: pre-aggregated campaign-level metrics
- `data_dictionary.csv`: column definitions
- `assets/`: synthetic creative images referenced by `creatives.csv`

## Dataset size
- Advertisers: 36
- Campaigns: 180
- Creatives: 1,080
- Daily rows: 192,315

## Join keys
- `advertisers.advertiser_id = campaigns.advertiser_id`
- `campaigns.campaign_id = creatives.campaign_id`
- `creatives.creative_id = creative_daily_country_os_stats.creative_id`
- `campaigns.campaign_id = creative_daily_country_os_stats.campaign_id`

## Notes
- All names, apps, assets and metrics are synthetic.
- The data contains **realistic patterns**, including:
  - different creative preferences by vertical,
  - different performance by country and OS,
  - creative fatigue over time,
  - uneven spend allocation across creatives.
- Some columns are engineered summaries meant to make the problem approachable in a hackathon.
- Hidden generator variables were intentionally excluded from the public files.

## Suggested student tasks
**Beginner**
- Rank creatives by performance within a campaign
- Compare winning traits by vertical

**Intermediate**
- Detect fatigue using the daily table
- Explain performance drops by feature group

**Advanced**
- Recommend the next creative variation to test for a campaign
- Build a small copilot that explains creative performance in plain English

## Known Quirks

**`fatigue_day` is only populated for fatigued creatives.**
Rows where `creative_status` is `top_performer`, `stable`, or `underperformer` have a blank `fatigue_day`.
Use `creative_status == "fatigued"` to filter for fatigued creatives — `fatigue_day.notna()` gives the same result but is redundant.

**Portfolio structure is perfectly uniform by design.**
Every advertiser has exactly 5 campaigns and every campaign has exactly 6 creatives.
Any "most active advertiser" or "biggest portfolio" analysis will return a tie across all advertisers — focus on performance metrics instead.

## Caveat
This dataset is designed for learning, prototyping and demos — not for benchmarking real production models.
