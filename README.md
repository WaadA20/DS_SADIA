# project

Predicting Hotel Booking Cancelation

Context

A significant number of hotel bookings are called-off due to cancellations or no- shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with. Such losses are particularly high on last-minute cancellations.
The new technologies involving online booking channels have dramatically changed customers’ booking possibilities and behavior. This adds a further dimension to the challenge of how hotels handle cancellations, which are no longer limited to traditional booking and guest characteristics.
The cancellation of bookings impact a hotel on various fronts:
• Lossofresources(revenue)whenthehotelcannotreselltheroom.
• Additionalcostsofdistributionchannelsbyincreasingcommissionsorpaying
for publicity to help sell these rooms.
• Loweringpriceslastminute,sothehotelcanresellaroom,resultingin
reducing the profit margin.
• Humanresourcestomakearrangementsfortheguests.

Objective:

The increasing number of cancellations calls for a Machine Learning based solution that can help in predicting which booking is likely to be canceled. Star Hotels Group they are facing problems with the high number of booking cancellations and have reached out to your firm for data-driven solutions. You as a data scientist have to analyze the data provided to find which factors have a high influence on booking cancellations, build a predictive model that can predict which booking is going to be canceled in advance, and help in formulating profitable policies for cancellations and refunds.

 Question/need
 1. What are the busiest months in the hotel?
2. Which market segment do most of the guests come from?
3. Many guests have special requirements when booking a hotel room. Do these
requirements affect booking cancellation?
4. What percentage of bookings are canceled?


 Data Description
The data contains the different attributes of customers' booking details. The detailed data dictionary given below.

Data Dictionary
• no_of_adults:Numberofadults
• no_of_children:NumberofChildren
• no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the
guest stayed or booked to stay at the hotel
• no_of_week_nights:Numberofweeknights(MondaytoFriday)theguest
stayed or booked to stay at the hotel
• type_of_meal_plan:Typeofmealplanbookedbythecustomer:
 • NotSelected–Nomealplanselected
• MealPlan1–Breakfast
• MealPlan2–Halfboard(breakfastandoneothermeal) • MealPlan3–Fullboard(breakfast,lunch,anddinner)
 • required_car_parking_space:Doesthecustomerrequireacarparkingspace?(0 - No, 1- Yes)
• room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by Star Hotels.
• lead_time:Numberofdaysbetweenthedateofbookingandthearrivaldate
• arrival_year:Yearofarrivaldate
• arrival_month:Monthofarrivaldate
 
• arrival_date:Dateofthemonth
• market_segment_type:Marketsegmentdesignation.
• repeated_guest:Isthecustomerarepeatedguest?(0-No,1-Yes)
• no_of_previous_cancellations: Number of previous bookings that were
canceled by the
• customerpriortothecurrentbooking
• no_of_previous_bookings_not_canceled: Number of previous bookings not
canceled by
• thecustomerpriortothecurrentbooking
• avg_price_per_room: Average price per day of the reservation; prices of the
rooms are
• dynamic.(ineuros)
• no_of_special_requests:Totalnumberofspecialrequestsmadebythecustomer
(e.g.
• highfloor,viewfromtheroom,etc)
• booking_status: Flag indicating if the booking was canceled or not. (targer
variable)
