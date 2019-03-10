#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Nodar Okroshiashvili'
SITENAME = 'Data Science Fabric'
SITEURL = '/'
SITESUBTITLE = u"Torture the data, and it will confess to anything. Ronald Coase"

SITE_DESCRIPTION = u'My name is Nodar Okroshiashvili \u2015 a data scientist who gets things done. I am Okroshiashvili at Github and @N_Okroshiashvil at twitter. I Stopped Dreaming and Started Doing.'


PATH = 'content'
TIMEZONE = 'Asia/Tbilisi'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None



# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)



# Social widget
SOCIAL_PROFILE_LABEL = 'Stay in Touch'

SOCIAL = (('linkedin-square', 'https://www.linkedin.com/in/nodar-okroshiashvili-a86571131/'),
          ('Github', 'https://github.com/Okroshiashvili'),
		  ('Twitter', 'https://twitter.com/N_Okroshiashvil'))

TWITTER_USERNAME = 'N_Okroshiashvil'

###       Custom Settings    ###


THEME = 'pelican-themes/elegant'

PLUGIN_PATHS = ['pelican-plugins']
PLUGINS = ["tipue_search",
            "render_math",
            'pelican-ipynb.markup',
            'sitemap',
            'neighbors',
            'share_post',
            'related_posts']


IGNORE_FILES = [".ipynb_checkpoints"]


SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.5,
        'indexes': 0.5,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}


STATIC_PATHS = ['theme/images', 'images']
USE_SHORTCUT_ICONS=True



DIRECT_TEMPLATES = (('index', 'tags', 'categories', 'archives', 'search', '404'))

TIPUE_SEARCH = True

TEMPLATE_PAGES = {
        'search.html': 'search.html',
        }



ARTICLE_PATHS = ['articles',]
ARTICLE_URL = 'articles/{category}/{slug}.html'
ARTICLE_SAVE_AS = 'articles/{category}/{slug}.html'


# If True, load unmodified content from cache
LOAD_CONTENT_CACHE = False


IPYNB_IGNORE_CSS = True



MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
        'markdown.extensions.toc' :{'permalink' : 'true'},

    },
    'output_format': 'html5',
}




OUTPUT_RETENTION = [".hg", ".git", "CNAME"]



RELATED_POSTS_LABEL = 'Keep Reading'
SHARE_POST_INTRO = 'Like this post? Share on:'
COMMENTS_INTRO = u'So what do you think? Did I miss something? Is any part unclear? Leave your comments below.'



# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True






LANDING_PAGE_ABOUT = {'title': 'Stop Dreaming - Start Doing',
        'details': """<div itemscope itemtype="http://schema.org/Person"><p>My name
        is <span itemprop="name">Nodar Okroshiashvili</span>.
       I am <a href="https://github.com/Okroshiashvili/" title="My Github
       profile" itemprop="url"><span itemprop="nickname">Okroshiashvili</span></a> at Github and <a
       href="https://twitter.com/N_Okroshiashvil/" title="My Twitter
       profile" itemprop="url">@N_Okroshiashvil</a> at twitter. You can also reach me via <a
       href="mailto:n.okroshiashvili@gmail.com" title="My email
       address" itemprop="email">email</a>.</p><p>I work for <a href="https://credo.ge/"
       title="Credo Bank" itemprop="affiliation">Credo Bank</a> which is a local commercial bank.
       I am a data scientist at Research and Development unit, which is under Marketing Department.
       Long story short, I model consumer behaviour, using various ML  techniques and based on 
       analysis create products and offer those products to consumer.</p><p> I hold BS in
       Business Administration, MA in Economics and last year started my PhD, again in Economics.
       </p><p>I try to help aspiring data scientist, not to make the same mistakes I did, 
       when starting transition from Economics to Data Science.</p><p>Besides programming, I do  
       programming, and besides all that I love fishing. My motto is: Do it, and do it right now.
       </p></div>"""}




















