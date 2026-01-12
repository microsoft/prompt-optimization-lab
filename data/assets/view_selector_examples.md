**Note:** 
 1. For questions related to historical investments, always use the **INVESTMENT_KPI_VW**.
 2. To retrieve Partner and active deals information, ALWAYS USE the **ACTIVE_DEAL_LIST_VW**
 3. **CRITICAL - MIC/ADIC Portfolio Queries**: When user asks about "MIC's exposure", "MIC's returns", "ADIC's exposure" etc., they are asking for PORTFOLIO-LEVEL aggregation (all of Mubadala's investments), NOT a specific platform. Use MIC_* views (e.g., MIC_BY_ASSET_CLASS_VW, MIC_BY_COUNTRY_VW, MIC_BY_REGION_VW) for these queries.
 4. **Platform vs Portfolio**: "ADIC - Platform" or "Private Equity Platform" are PLATFORM names. "MIC" or "ADIC" alone refer to the PORTFOLIO level.
# View Selector Examples

These examples help the View Selection Agent understand how to choose appropriate database views based on user questions and resolved entities.

## Example 1: Portfolio-Level Query with Investment Class Filter
**Question:** At Group level, what is the recent ITD_RETURN for Public Portfolio?

**Resolved Entities:**
```json
{"INVESTMENT_CLASS": "Public Portfolio"}
```

**Process:**
1. User asks for Group level data with INVESTMENT_CLASS filter
2. "Group level" indicates portfolio-level aggregation (MIC portfolio)
3. Need a view that aggregates by INVESTMENT_CLASS and has ITD_RETURN
4. Let me get the schema: get_entity_schema("MIC_BY_INVESTMENT_CLASS_VW")
5. This view has INVESTMENT_CLASS, ITD_RETURN, and AS_ON_DATE_ID columns - perfect match

**Answer:** MIC_BY_INVESTMENT_CLASS_VW
**Note:** For portfolio-level queries with investment class breakdown, use MIC_BY_INVESTMENT_CLASS_VW which pre-aggregates data at the portfolio level.

## Example 2: Active Deals with Sector Filter
**Question:** Show me all active deals in the Technology sector

**Resolved Entities:**
```json
{"GICS_SECTOR": "Technology"}
```

**Process:**
1. User asks for "active deals" which requires deal-level information
2. Need a view that contains deal status and sector information
3. Let me get the schema: get_entity_schema("ACTIVE_DEAL_LIST_VW")
4. After examining columns, it has DEAL_STATUS, GICS_SECTOR, and deal identifiers - exactly what's needed

**Answer:** ACTIVE_DEAL_LIST_VW
**Note:** For active deal queries with sector filtering, use ACTIVE_DEAL_LIST_VW which contains current deal status and GICS sector information.

## Example 3: Portfolio-Level Exposure by Country and Asset Class
**Question:** What is MIC's current exposure in the USA in Private Equity asset class?

**Resolved Entities:**
```json
{"ASSET_CLASS": "Private Equity"}
```

**Process:**
1. "MIC's exposure" means PORTFOLIO-LEVEL aggregation, NOT platform filter
2. Need exposure data filtered by country and asset class
3. Since this requires granular filtering by multiple dimensions, INVESTMENT_KPI_VW is appropriate
4. Let me get the schema: get_entity_schema("INVESTMENT_KPI_VW")
5. Has UNREALIZED_VALUE (exposure), COUNTRY_ISO_CODE2/3 (USA), and ASSET_CLASS (Private Equity)

**Answer:** INVESTMENT_KPI_VW
**Note:** For portfolio-level queries requiring multiple dimensional filters (country + asset class), use INVESTMENT_KPI_VW. Do NOT add PLATFORM filter for "MIC" - it refers to portfolio level.

## Example 4: Portfolio-Level Aggregation by Region
**Question:** Report Group exposure by region

**Resolved Entities:**
```json
{}
```

**Process:**
1. "Group exposure by region" indicates portfolio-level aggregation with regional breakdown
2. Need a view that pre-aggregates exposure data by region
3. Let me get the schema: get_entity_schema("MIC_BY_REGION_VW")
4. This view has REGION, UNREALIZED_VALUE (exposure), and AS_ON_DATE_ID - perfect for regional aggregation

**Answer:** MIC_BY_REGION_VW
**Note:** For portfolio-level regional exposure queries, use MIC_BY_REGION_VW which pre-aggregates exposure by region at the portfolio level.

## Example 5: Granular Investment Performance
**Question:** What is the cash deployments for CityFibre UK YTD?

**Resolved Entities:**
```json
{}
```
**Process:**
1. User asks for specific investment cash deployments (deployment = investment)
2. Requires granular, deal-level data for a specific project
3. Let me get the schema: get_entity_schema("INVESTMENT_KPI_VW")
4. Has ITD_CASH_INVESTED, YTD_CASH_INVESTED, PROJECT_NAME, and AS_ON_DATE_ID columns needed

**Answer:** INVESTMENT_KPI_VW
**Note:** For granular investment-level queries, use INVESTMENT_KPI_VW which contains detailed cash flow information by project.

