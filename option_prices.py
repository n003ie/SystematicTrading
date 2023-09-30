import blpapi

def get_bond_option_data(option_tickers, field_list):
    options_data = {}

    sessionOptions = blpapi.SessionOptions()
    sessionOptions.setServerHost("localhost")
    sessionOptions.setServerPort(8194)  # This is the default port for Bloomberg's API

    session = blpapi.Session(sessionOptions)
    if not session.start():
        print("Failed to start session.")
        return

    try:
        service = session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")

        for option_ticker in option_tickers:
            request.append("securities", option_ticker)

        for field in field_list:
            request.append("fields", field)

        session.sendRequest(request)

        while True:
            event = session.nextEvent(500)  # 500 milliseconds timeout
            if event.eventType() == blpapi.Event.RESPONSE:
                for msg in event:
                    security_data = msg.getElement("securityData")
                    for sec_data in security_data.values():
                        option_ticker = sec_data.getElementAsString("security")
                        field_data = sec_data.getElement("fieldData")

                        option_info = {}
                        for field in field_list:
                            value = field_data.getElement(field).getValue()
                            option_info[field] = value

                        options_data[option_ticker] = option_info
            elif event.eventType() == blpapi.Event.RESPONSE_TIMEOUT:
                break
            else:
                print("Non-Response event received.")

    finally:
        session.stop()

    return options_data


def main():
    option_tickers = ["TYX 09/23 P150", "TYX 09/23 P155", "TYX 09/23 P160"]
    field_list = ["PX_LAST", "OPT_EXPIRE_DT", "STRIKE_PX", "IVOL_MID", "OPT_DELTA", "OPT_VEGA", "OPT_THETA",
                  "OPT_GAMMA", "OPT_RHO"]

    options_data = get_bond_option_data(option_tickers, field_list)

    for option_ticker, option_info in options_data.items():
        print(f"Option: {option_ticker}")
        print(f"Last Price: {option_info['PX_LAST']}")
        print(f"Expiry Date: {option_info['OPT_EXPIRE_DT']}")
        print(f"Strike Price: {option_info['STRIKE_PX']}")
        print(f"Implied Volatility: {option_info['IVOL_MID']}")
        print(f"Delta: {option_info['OPT_DELTA']}")
        print(f"Vega: {option_info['OPT_VEGA']}")
        print(f"Theta: {option_info['OPT_THETA']}")
        print(f"Gamma: {option_info['OPT_GAMMA']}")
        print(f"Rho: {option_info['OPT_RHO']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
