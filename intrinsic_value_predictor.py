import numpy as np

def calculate_intrinsic_value(cash_flows, growth_rate, discount_rate, terminal_growth_rate):
    """
    Calculate the intrinsic value of a stock using the Discounted Cash Flow (DCF) method.

    :param cash_flows: List of projected future cash flows (for example, over the next 5 years).
    :param growth_rate: Annual growth rate of the cash flows beyond the projection period.
    :param discount_rate: Discount rate to discount future cash flows to their present value.
    :param terminal_growth_rate: Growth rate to estimate the terminal value.
    :return: Intrinsic value of the stock.
    """
    # Calculate the present value of projected cash flows
    present_value_of_cash_flows = sum(cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows))

    # Calculate the terminal value
    last_cash_flow = cash_flows[-1]
    terminal_value = last_cash_flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value_of_terminal_value = terminal_value / (1 + discount_rate) ** len(cash_flows)

    # Calculate the intrinsic value
    intrinsic_value = present_value_of_cash_flows + present_value_of_terminal_value

    return intrinsic_value

# Example usage
last_cash_flow = 1642.5e6
growth_rate = 0.10  # 10% growth rate
n_years = 10
number_of_shares = 2.64e6

projected_cash_flows = [last_cash_flow * (1 + growth_rate) ** i for i in range(1, n_years + 1)] 
discount_rate = 0.12  # 12% discount rate
terminal_growth_rate = 0.03  # 3% terminal growth rate

intrinsic_value = calculate_intrinsic_value(projected_cash_flows, growth_rate, discount_rate, terminal_growth_rate)
print(f"Intrinsic Value of the stock: ${intrinsic_value/number_of_shares:.2f}")