#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Nodar Okroshiashvili'
SITENAME = 'Data Science Fabric'
SITEURL = ""
SITESUBTITLE = u"Torture the data, and it will confess to anything. Ronald Coase"

PATH = 'content'


# Regional settings
TIMEZONE = 'Asia/Tbilisi'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None



# Plugins and extensions

PLUGIN_PATHS = ['pelican-plugins']

PLUGINS = ["tipue_search",
            "render_math",
            'pelican-ipynb.markup',
            'sitemap',
            'neighbors',
            'share_post',
            'related_posts',
            'latex']


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


MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.admonition': {},
        'markdown.extensions.codehilite': {'css_class': 'highlight'},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
        'markdown.extensions.toc' :{'permalink' : 'true'},

    },
    'output_format': 'html5',
}

IPYNB_IGNORE_CSS = True

OUTPUT_RETENTION = [".hg", ".git", "CNAME"]

IGNORE_FILES = [".ipynb_checkpoints"]

USE_SHORTCUT_ICONS=True










# Blogroll
LINKS = (('Pelican', 'http://getpelican.com/'),
         ('Python.org', 'http://python.org/'),
         ('Jinja2', 'http://jinja.pocoo.org/'),
         ('You can modify those links in your config file', '#'),)



# Social
SOCIAL_PROFILE_LABEL = 'Stay in Touch'

SOCIAL = (('linkedin', 'https://www.linkedin.com/in/nodar-okroshiashvili/'),
          ('Github', 'https://github.com/Okroshiashvili'),
		  ('Twitter', 'https://twitter.com/N_Okroshiashvil'))

TWITTER_USERNAME = 'N_Okroshiashvil'



# Appearance
THEME = 'pelican-themes/elegant'

TYPOGRIFY = True

STATIC_PATHS = ['theme/images', 'images', 'extra']
EXTRA_PATH_METADATA = {
    'extra/robots.txt': {'path': 'robots.txt'}}



DIRECT_TEMPLATES = ["index", "tags", "categories", "archives", "search", "404"]

TIPUE_SEARCH = True

TEMPLATE_PAGES = {
        'search.html': 'search.html',
        }

ARTICLE_PATHS = ['articles',]
ARTICLE_URL = '{slug}'
ARTICLE_SAVE_AS = '{slug}.html'

TAGS_URL = "tags"
CATEGORIES_URL = "categories"
ARCHIVES_URL = "archives"

PAGE_URL = "{slug}"
PAGE_SAVE_AS = "{slug}.html"

TAG_SAVE_AS = ""
AUTHOR_SAVE_AS = ""
CATEGORY_SAVE_AS = ""

SEARCH_URL = "search"

RELATED_POSTS_LABEL = 'Keep Reading'
SHARE_POST_INTRO = 'Like this post? Share on:'
COMMENTS_INTRO = u'So what do you think? Did I miss something? Is any part unclear? Leave your comments below.'


# If True, load unmodified content from cache
LOAD_CONTENT_CACHE = False



# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True




# SEO
SITE_DESCRIPTION = (
    "Blog about data science, mathematics and Python"
)


LANDING_PAGE_TITLE = "Unstructured Thoughts"


