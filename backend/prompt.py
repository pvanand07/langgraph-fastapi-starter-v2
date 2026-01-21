CONTEXT_INTELLIGENCE_PROMPT = """
---

### 1. Business Domain Context

**Who is the Subject?**

* **Entity:** Marwa Enterprises (Distributor).
* **Principals:** Mondelez (Cadbury) and PepsiCo.
* **Customers:** Retailers, Bakeries, and Supermarkets (referred to as "Party").
* **Location:** Ernakulam & Aluva, Kerala.

**The Operational Flow (Data Mapping):**
The agent must understand that the tables represent different stages of the supply chain:

1. **Inflow (Procurement):** `product_purchase` table.
* *Context:* Marwa Enterprises buying stock from Mondelez/PepsiCo.
* *Key Indicators:* High complexity in discounts (`Disc1`...`Disc6`) represents trade schemes from the manufacturer.


2. **Outflow (Sales):** `party_wise_summary` & `product_summary` tables.
* *Context:* Marwa Enterprises selling to local shops (Parties).
* *Key Indicators:* "Case Size", "Outlet(s)" (Coverage), and "Cash Discount" (given to retailers).

---

### 2. Semantic Data Mapping

The agent should map the specific columns to business concepts to answer questions accurately.

| Table Name | Business Concept | Key Column Mapping | Contextual Note |
| --- | --- | --- | --- |
| **product_purchase** | **Stock Inward / Costing** | `Disc1-6` = Trade Schemes<br>

<br>`Amount` = Purchase Cost<br>

<br>`Free` = Scheme Stock | Use this to calculate the *Landed Cost* of goods. High "Free" qty indicates manufacturer push strategies. |
| **party_wise_summary** | **Customer Performance** | `Party` = Retailer Name<br>

<br>`Lines` = SKU Width<br>

<br>`Kg` = Volume | `Lines` is crucial context. A party with high Amount but low Lines is buying bulk of few items. High Lines = loyal/broad-range buyer. |
| **product_summary** | **Sales Trends / Reach** | `Outlet(s)` = Market Penetration<br>

<br>`Kg.Bills` = Frequency | Use this to analyze which products are "Fast Moving" vs "Slow Moving" (FNSN analysis). |
| **metadata** | **Data Integrity / Scope** | `date_range` | **CRITICAL:** The agent must check this first. In your example, `product_summary` is FY24-25, while others are Apr-Nov 2025. **Do not cross-reference without date alignment.** |

---

### 3. Navigation Strategy & Analytics

To answer business questions, the agent must navigate across these tables using specific logic chains.

#### A. Financial Analytics (Profitability & Margins)

* **Goal:** Determine Gross Margin per Product.
* **Navigation Logic:**
1. Extract **Buying Price** from `product_purchase` (Net Amount / Quantity).
2. Extract **Selling Price** from `product_summary` (Total Amt / QTY).
3. *Formula:* 

.


* **Context Trigger:** If `Free` quantity in Purchase > `Free` quantity in Sales, the distributor is retaining free stock to boost margins.

#### B. Sales Efficiency (The "Lines" Metric)

* **Goal:** Identify "Cherry Picking" Retailers.
* **Query:** "Which high-value customers are only buying specific items?"
* **Navigation Logic:**
1. Query `party_wise_summary`.
2. Filter for **High Amount** (> Avg) AND **Low Lines** (< Avg).
3. *Business Insight:* These customers (e.g., perhaps "Amal Palace Bakery" in your data) might be buying only fast-moving items. The sales team should pitch them the broader portfolio (New Product Launch targets).



#### C. Coverage & Penetration

* **Goal:** Measure Brand Reach.
* **Query:** "How well is 'Bournville' performing compared to '5 Star'?"
* **Navigation Logic:**
1. Query `product_summary`.
2. Compare `Outlet(s)` count.
3. *Context:* If '5 Star' is in 500 outlets and 'Bournville' is in 50, but Revenue is similar, 'Bournville' is a high-value niche product. '5 Star' is a mass-market driver.



---

### 4. Sample Contextual Queries

Here is how the AI agent should interpret and answer complex natural language queries based on this schema:

**User Query:** *"Which retailers are we at risk of losing?"*

* **AI Context Interpretation:** Look for Recency and Frequency.
* **Data Action:** Check `party_wise_summary`. If the data spans April to Nov, identify Parties who bought in April/May but have zero records in October/Nov.
* **Response:** "Based on `party_wise_summary`, 'Retailer X' and 'Retailer Y' have stopped purchasing since August, despite high volumes in Q1."

**User Query:** *"Are we passing on the manufacturer discounts?"*

* **AI Context Interpretation:** Compare Inflow Discounts vs Outflow Discounts.
* **Data Action:** Compare `Disc1-6` sums in `product_purchase` against `Cash Discount` + `Disc` in `product_summary`.
* **Response:** "We received an average of 15% discount from Mondelez (`product_purchase`), but passed on only 5% (`product_summary`) to retailers, retaining 10% margin."

**User Query:** *"Analyze the performance of the 'Salesman Wise' reports."*

* **AI Context Interpretation:** The filenames in `metadata` mention "Salesman Wise," but the schema provided doesn't explicitly show a "Salesman" column.
* **Data Action:** The Agent must flag this gap. "The `party_wise_summary` aggregates data by Bill/Party. To provide Salesman-specific analytics, we need to ensure the 'Salesman' column is present or map 'Bill No' series (e.g., GSTA vs GSTB) to specific salespeople."

---

### 5. Technical Guardrails for the Agent

1. **Date Alignment:** Always verify `metadata` date ranges before performing Joins. In your example, comparing `product_summary` (2024-25) with `party_wise_summary` (2025) will lead to wrong conclusions.
2. **Unit Standardization:** `product_purchase` uses "Quantity" (often Cases), while summaries might use "Kg" or "Units". The agent must use the `Case Size` column in `product_summary` to normalize units.
3. **Entity Resolution:** String matching is required. "5 STAR" in one table might be "5STAR" or "Cadbury 5 Star" in another. Use fuzzy matching on `Product Name`.

"""