import blpapi
import pandas as pd


def retrieve_intraday_bond_future_data(ticker, start_date, end_date, interval=1, fields=None):
    # Create a SessionOptions object to manage session settings
    if fields is None:
        fields = ["Open", "High", "Low", "Close", "Volume"]

    session_options = blpapi.SessionOptions()
    session_options.setServerHost("localhost")  # Bloomberg API server host

    # Create a Session object
    session = blpapi.Session(session_options)

    try:
        # Start the session
        if not session.start():
            print("Failed to start session.")
            return None

        # Open a service to get intraday data (replace with the appropriate service name)
        if not session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata service.")
            return None

        # Create a request for intraday bond future data (replace with the appropriate security)
        request = session.createRequest("IntradayBarRequest")
        request.getElement("security").setValue(ticker)  # Replace with your bond future security
        request.getElement("eventType").setValue("TRADE")
        request.set("startDateTime", f"{start_date}T00:00:00.000Z")
        request.set("endDateTime", f"{end_date}T23:59:59.999Z")
        request.set("interval", interval)  # 1-minute interval

        # Send the request
        session.sendRequest(request)

        # Create an empty DataFrame to store the intraday data
        data_df = pd.DataFrame(columns=["Date", "Time"] + fields)

        # Wait for responses
        while True:
            event = session.nextEvent()
            if event.eventType() == blpapi.Event.RESPONSE:
                data_df = process_response(event, data_df)
                break

    finally:
        # Stop the session
        session.stop()

    return data_df


def process_response(event, data_df):
    for msg in event:
        if msg.hasElement("bar_data"):
            bar_data = msg.getElement("bar_data")
            for i in range(bar_data.numValues()):
                bar = bar_data.getValueAsElement(i)
                date = bar.getElementAsDatetime("time").date()
                time = bar.getElementAsDatetime("time").time()
                open_price = bar.getElementAsFloat("open")
                high_price = bar.getElementAsFloat("high")
                low_price = bar.getElementAsFloat("low")
                close_price = bar.getElementAsFloat("close")
                volume = bar.getElementAsFloat("volume")

                data_df = data_df.append({
                                        "Date": date,
                                        "Time": time,
                                        "Open": open_price,
                                        "High": high_price,
                                        "Low": low_price,
                                        "Close": close_price,
                                        "Volume": volume
                                        }, ignore_index=True)

    return data_df


if __name__ == "__main__":
    start_date = "YYYY-MM-DD"
    end_date = "YYYY-MM-DD"
    ticker = 'TYA Comdty'
    intraday_data = retrieve_intraday_bond_future_data(ticker, start_date, end_date)

    if intraday_data is not None:
        print(intraday_data)
