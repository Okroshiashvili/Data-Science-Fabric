#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Nodar Okroshiashvili'
SITENAME = 'Data Science Fabric'
SITEURL = ''


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
