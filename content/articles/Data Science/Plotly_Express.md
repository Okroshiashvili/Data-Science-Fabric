Title: Simple Interactive Data Visualization with Plotly Express
Date: 2019-03-22 12:14
Modified: 2020-03-26 20:53
Category: Data Science
Tags: Plotly, Data Visualization
Keywords: plotly, plotly express, python, data visualization, interactive data visualization, plotly dashboards
Author: Nodar Okroshiashvili
Summary: Interactive data visualization with plotly express

I fairly believe that, if someone is good at data visualization he/she at least has heard about Hans Rosling. 
One of the most influential data storyteller, and honestly because of the [talk](https://www.ted.com/talks/hans_rosling_shows_the_best_stats_you_ve_ever_seen) he gave at TED, 
it influenced me too much that I decided to learn data visualization. Since then, I had some progress as well some stagnation but the most noteworthy thing here is that 
I got my hands dirty with great data visualization tools such as [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [bokeh](https://bokeh.pydata.org/en/latest/), [altair](https://altair-viz.github.io/), and [plotly](https://plot.ly/python/). 


All of these data visualization tools are great in its own niche and describing 
the pros and cons of each other is the matter of another separate blog. 
Having in mind that today's world is more interactive and most data viz wizards 
like "one line" solutions(almost true). Plotly made a huge improvement in this direction. 
Two days ago they introduced [Plotly Express](https://twitter.com/plotlygraphs/status/1108395986626494467) 
which is a new high-level Python visualization library. Long story short it is the wrapper for [Plotly.py](https://plot.ly/python/) that makes building complex charts simple.

> Note that, at the moment of writing ```Plotly Express``` was seperately installable. 
> However, now  it's part of ```Plotly``` and can be imported by calling ```import plotly.express as px```. 


Developers main aim for building plotly express:

> Our main goal with Plotly Express was to make it easier to use Plotly.py for exploration and rapid iteration.


Let's go through some example and see how plotly express works.



```python
import plotly.express as px
import plotly.offline as pyo
```




```python
# Plotly Express has build in datasets. You can load this datasets by uncommenting below.
# For expositional purposes, I use well-known gapminder data

gapminder = px.data.gapminder()

# iris = px.data.iris()

# tips = px.data.tips()

# election = px.data.election()

# wind = px.data.wind()

# carshare = px.data.carshare()
```





```python
# Extract data for 2007

gapminder2007 = gapminder.query("year == 2007")

gapminder2007.head()
```

```
        country continent  year  lifeExp       pop     gdpPercap  iso_alpha  iso_num
59    Argentina  Americas  2007   75.320  40301927  12779.379640  ARG        4
11  Afghanistan      Asia  2007   43.828  31889923    974.580338  AFG        8
23      Albania    Europe  2007   76.423   3600523   5937.029526  ALB        12
35      Algeria    Africa  2007   72.301  33333216   6223.367465  DZA        24
47       Angola    Africa  2007   42.731  12420476   4797.231267  AGO        32
```





```python
# Simple scatter plot.

# GDP Per Capita vs Life Expectancy

px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy'})
```



<iframe src="../../images/first_plot.html", width="100%", height="600px"></iframe>


Pelican produces static html files. So, plotly interactive charts do not show up. 
Firstly, I converted plotly chart into standalone html file and then added by using IFrame tag. 
That's why it is here.

Each point represents country with its own continent. To see continents we can color each dot by continent name.



```python
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color='continent',
                        labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy'})
```



<iframe src='../../images/second_plot.html', width="100%", height="600px"></iframe>


And, what if we want to adjust each point size as it to be the country population size? That's should not be a problem.



```python
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color='continent', 
                        size='pop', size_max=60, 
                        labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy'})
```



<iframe src='../../images/third_plot.html', width="100%", height="600px"></iframe>


If we hover-over the mouse to each data point the point description will appear. 
It shows continent name, x and y coordinates, and population size. 
As we already know each point is a country but it's not properly shown in the description. We can achieve this easily.




```python
px.scatter(gapminder2007, x="gdpPercap", y="lifeExp", color='continent', 
                        size='pop', size_max=60, hover_name='country', 
                        labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy'})
```



<iframe src='../../images/fourth_plot.html', width="100%", height="600px"></iframe>


I wrote initial command and at each step added two extra arguments. That's the advantage of Plotly Express.

Now, what if want to see how this chart evolves over time. Simply, we need to see what happened before 2007 and make it more interactive.




```python
px.scatter(gapminder, x="gdpPercap", y="lifeExp", color='continent', 
                    size='pop', size_max=60, hover_name='country', 
                    animation_frame='year', animation_group='country', 
                    log_x = True, range_x = [100, 100000], range_y = [25, 90], 
                    labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy', 'pop':'Population'})
```



<iframe src='../../images/fifth_plot.html', width="100%", height="600px"></iframe>


That's not all. Let plot same data on a map.



```python
px.choropleth(gapminder, locations='iso_alpha', color='lifeExp', 
                        hover_name='country', animation_frame='year', 
                        color_continuous_scale = px.colors.sequential.Plasma, projection='natural earth', 
                        labels={'lifeExp':'Life Expectancy'})
```



<iframe src='../../images/sixth_plot.html', width="100%", height="600px"></iframe>


That was good, but wait. What if you don't like jupyter?, [trust me there are such guys](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit#slide=id.g362da58057_0_1)  

That's not a problem. Combine Plotly-Express with Plotly and plot offline to make standalone html file.



```python
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", color='continent', 
                            size='pop', size_max=60, hover_name='country', 
                            animation_frame='year', animation_group='country', 
                            log_x = True, range_x = [100, 100000], range_y = [25, 90], 
                            labels={'gdpPercap':'GDP Per Capita', 'lifeExp':'Life Expectancy', 'pop':'Population'})
```




```python
# Save figure object as html file

pyo.plot(fig, filename='gapminder.html')
```






After glorifying plotly express, it is natural to find some drawbacks. Has plotly express some limitations? 
Particularly, how much data can we put into it? According to them, there is no hard limit for data set size 
but it is preferable to use 1000 points for some chart types and make some input parameter adjustments, 
while other chart types easily handle more points. I'm still trying to figure out the mechanics of plotly 
express and dig deeper, however, it won't be a bad idea to check more, about plotly express [here](https://medium.com/@plotlygraphs/introducing-plotly-express-808df010143d).


### References

- [Medium Blog](https://medium.com/@plotlygraphs/introducing-plotly-express-808df010143d)
- [Plotly Express](https://plotly.github.io/plotly_express/plotly_express/)
- [Example Gallery](https://plotly.github.io/plotly_express/)
