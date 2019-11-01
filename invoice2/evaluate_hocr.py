from __future__ import print_function
import argparse
import re
import sys
import cv2
import numpy as np
from lxml import html
from multi_key_dict import multi_key_dict
# import the necessary packages
from collections import namedtuple
import json
testing = False
import locale
# locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 
from get_table_coords import get_xs, show_wait_destroy
import pandas as pd
import pdb
from pathlib import Path, PureWindowsPath
from config import *
glob = globals()
# # This is equivalent to "from a.b import myfile"
# myfile = __import__("a.b.myfile", fromlist="a.b")
# also_myfile = importlib.import_module("a.b.myfile", "a.b")
# output = myfile.my_method()


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def get_bbox_by_word_id(word_id):
	info =root.get_element_by_id(word_id).get('title')   
	info = info.split(";")
	if len(info)==3:
		bbox,conf,size = info
	if len(info)==4:
		bbox, conf, font, size = info
	bbox = bbox.split(" ")[1:]
	bbox = [int(x) for x in bbox]
	return bbox
def get_fsize_by_word_span(span):
	info = span.get('title')
	info = info.split(";")
	if len(info)==3:
		bbox,conf,size = info
	if len(info)==4:
		bbox, conf, font, size = info
	size = size.split(" ")[-1]
	size = int(size)
	return size
	
def get_line_fsize(ocr_line):
	sizes =[get_fsize_by_word_span(x) for x in ocr_line]
	id = ocr_line.get('id')
	if sizes:
		return (id, sum(sizes)/len(sizes))
	else:
		return (id,0)


def get_bbox_by_line_id(line_id):
	info=root.get_element_by_id(line_id).get('title')
	bbox = get_bbox_by_title(info)
	return bbox


def get_right_boxes(bbox):
	(x1, y1, x2, y2) = bbox
	x1_min = x2
	x1_max = 99999
	x2_min = x2-100
	x2_max = 99999
	y1_min = y1-100
	y1_max = y2-100
	y2_min = 0
	y2_max = 99999
	min_bbox = [x1_min, y1_min, x2_min, y2_min]
	max_bbox = [x1_max, y1_max, x2_max, y2_max]
	# word_spans = root.find_class('ocrx_word')
	# word_ids = [get_id_by_span(word_span) for word_span in word_spans]
	# word_bboxes = [get_bbox_by_word_id(word_id) for word_id in word_ids]
	rel_bboxes = [(word_ids[i],bbox) for i,bbox in enumerate(word_bboxes) if bbox_in_limits(bbox, min_bbox, max_bbox)]
	return rel_bboxes

def get_below_boxes(bbox, strict = True, ymax=99999, text=None, extract=None):
	(x1, y1, x2, y2) = bbox
	x1_min = x1-200
	x1_max = x2
	if not strict:
		x2_min = 0
		x2_max = 99999
	else:
		x2_min = x2-100
		x2_max = x2+100
	y1_min = y2-100
	y2_min = y1_min
	y1_max = 99999
	y2_max = min(99999, ymax)
	if text=='Amount':
		if extract=='Totals':
			# print()
			y1_min = y1-200
			y2_max = 99999
	if text == 'Invoice':
		y2_max = y2+600
	min_bbox = [x1_min, y1_min, x2_min, y2_min]
	max_bbox = [x1_max, y1_max, x2_max, y2_max]
	# word_spans = root.find_class('ocrx_word')
	# word_ids = [get_id_by_span(word_span) for word_span in word_spans]
	# word_bboxes = [get_bbox_by_word_id(word_id) for word_id in word_ids]
	rel_bboxes = [(word_ids[i],bbox) for i,bbox in enumerate(word_bboxes) if bbox_in_limits(bbox, min_bbox, max_bbox)]
	return rel_bboxes

def get_spans_by_text(text):
	return [x for x in root.iterdescendants() if str(text) in str(x.text)]
def get_id_by_span(word_span):
	# assert word_span.get('class') == 'ocrx_word'
	return word_span.get('id')

def get_bboxes_by_text(text, id=None):
	spans = get_spans_by_text(text)
	if id is None:
		bboxes = [get_bbox_by_word_id(get_id_by_span(span)) for span in spans]
	if id is not None:
		bboxes = [get_bbox_by_word_id(get_id_by_span(span)) for span in spans if get_id_by_span(span)==id]
	return bboxes
def bbox_in_limits(bbox, min_bbox, max_bbox):
	for i in range(4):
		if not min_bbox[i]<bbox[i]<max_bbox[i]:
			return False
	return True

def getTextAfterEncoding(x):
	try:
		bts = x.text.encode('cp1252')
		bts = str(bts, 'cp1252', 'ignore')
		# bts = unicode(bts, errors='replace')
	except Exception as e:
		print('Error: ',e)
		bts = ''
		
	return bts.strip()

def get_text_by_line_id(line_id):
	line = root.get_element_by_id(line_id)
	texts = [x.text.strip() for x in line.iterchildren() if x.text]
	texts = ' '.join(texts)
	return texts.strip()



def bbox_within(bbox, bbox_out):
	out_x1, out_y1, out_x2, out_y2 = bbox_out
	dx = out_x2-out_x1
	dy = out_y2 - out_y1
	# out_x1 -= 0.5*dx
	# out_y1 -= 0.5*dy
	# out_x2 += 0.5*dx
	# out_y2 += 0.5*dy
	out_x1 -= 0.02*dx
	out_y1 -= 0.3*dy
	out_x2 += 0.02*dx
	out_y2 += 0.3*dy
	if out_x1 < bbox[0] < out_x2:
		if out_y1 < bbox[1] < out_y2:
			if out_x1 < bbox[2] < out_x2:
				if out_y1 < bbox[3] < out_y2:
					return True
	return False
def get_close_right_rank(bbox1, bbox2):
	#bbox1_x2 closest to bbox2_x1
	_,_,x2,_ = bbox1
	x1,_,_,_ = bbox2
	return (x1-x2)
def get_close_below_rank(bbox1, bbox2):
	#bbox1_y2 closest to bbox2_y1
	_,_,_,y2 = bbox1
	_,y1,_,_ = bbox2
	return (y1-y2)

def get_bbox_by_title(title):
	info = title.split(";")
	bbox=info[0].split(' ')[1:]
	bbox = [int(x) for x in bbox]
	return bbox
	
def get_adj_by_text(text, direction = 'default', strict = False, ymax= 99999, id = None):
	all_occur = get_bboxes_by_text(text, id)
	cumulative_output=[]
	for _,o in enumerate(all_occur):
		if direction == 'below':
			o[1], o[2] = get_enclosing_xrange(o[1], o[2], table_xs)
			rel_boxes = get_below_boxes(o, strict = strict, ymax=ymax, text=text)
			box_ranks = [get_close_below_rank(o,r) for _,r in rel_boxes]
			box_sorted = [x for _,x in sorted(zip(box_ranks,rel_boxes),reverse=False)]
			solutions = [get_text_by_id(box[0]) for box in box_sorted]
			# print(occur_n, text,direction)
			# print("possible=",solutions)
			output = filter_by_expected_value(solutions,text)
			# print("after filter=",output)
			# print("++++++++++++++++++++++")
		elif direction == 'right':
			rel_boxes = get_right_boxes(o)
			box_ranks = [get_close_below_rank(o,r) for _,r in rel_boxes]
			box_sorted = [x for _,x in sorted(zip(box_ranks,rel_boxes),reverse=False)]
			solutions = [get_text_by_id(box[0]) for box in box_sorted]
			# print(occur_n, text,direction)
			# print("possible=",solutions)
			output = filter_by_expected_value(solutions,text)
			# print("after filter=",output)
			# print("++++++++++++++++++++++")
		elif direction == 'default':
			rel_boxes_below = get_below_boxes(o, strict = strict, ymax=ymax, text=text)
			box_ranks = [get_close_below_rank(o,r) for _,r in rel_boxes_below]
			box_below_sorted = [x for _,x in sorted(zip(box_ranks,rel_boxes_below),reverse=False)]
			rel_boxes_right = get_right_boxes(o)
			box_ranks = [get_close_right_rank(o,r) for _,r in rel_boxes_right]
			box_right_sorted = [x for _,x in sorted(zip(box_ranks,rel_boxes_right),reverse=False)]
			solutions_below = [get_text_by_id(box[0]) for box in box_below_sorted]
			solutions_right = [get_text_by_id(box[0]) for box in box_right_sorted]
			output_below = filter_by_expected_value(solutions_below,text)
			output_right = filter_by_expected_value(solutions_right,text)
			# if testing:
			# 	print(text, output_below)
			# 	print(text, output_right)
			output = output_below
			output.extend(output_right)
		cumulative_output.extend(output)
	# if direction == 'below':
	# 	cumulative_output = output
	return cumulative_output

def isAmount(x):
	strs = [x for x in x if x not in [',','.','+','-','(',')']]
	if len(strs)>0:
		isNum = sum([not str(x).isdigit() for x in strs])
	else:
		isNum = 1
	return isNum==0

def filter_by_expected_value(solutions, text):
	# Remove garbage; should contain alphanumeric
	solutions = [ s for s in solutions if s]
	# solutions = [x for x in solutions if re.search('\w',x) and len(x)>1]
	if text=='Invoice':
		numbers = [sum(c.isdigit() for c in s) for s in solutions]
		words   = [sum(c.isalpha() for c in s) for s in solutions]
		solutions = [x for i,x in enumerate(solutions) if numbers[i]>words[i]]
		solutions = [x for x in solutions if re.match("\d{1,2}[\-|\/]\w{2,3}[\-|\/]20\d{2}",x) is None]
	elif text=='GST':
		solutions = [x for x in solutions if re.match("\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}",x) is not None]
	elif text=='Date':
		# numbers = [sum(c.isdigit() for c in s) for s in solutions]
		# words   = [sum(c.isalpha() for c in s) for s in solutions]
		# contains_dot = ['.' in x for x in solutions]
		# solutions = [x for i,x in enumerate(solutions) if numbers[i]>words[i] and not contains_dot[i]] 
		solutions = [x for x in solutions if re.match("\d{1,2}[\-|\/]\w{2,3}[\-|\/]20\d{2}",x) is not None]
	elif text=='Quantity':
		solutions = [x for x in solutions if x.isdigit()]
	elif text=='Amount':
		solutions = [x for x in solutions if isAmount(x)]
	elif text=='Percentage':
		if not text.find('%') == -1:
			solutions = [x for x in solutions if not x == '%']
		solutions = [x for x in solutions if isAmount(x)]
	return solutions

def get_column_by_text(text):
	all_occur = get_bboxes_by_text(text)
	if len(all_occur)>0:
		for o in all_occur:
			below_boxes = get_below_boxes(o)
			# print(text)
			below_ids = [x[0] for x in below_boxes]
			below_texts = [get_text_by_id(x) for x in below_ids]
			# print(below_texts)
			# print("+++++++++++++++++")
		return list(zip(below_ids, below_texts))
	else:
		return []



def get_column_by_id(id, strict = True):
	occur = root.get_element_by_id(id)
	info = occur.get('title')
	bbox = get_bbox_by_title(info)
	bbox[0], bbox[2] = get_enclosing_xrange(bbox[0], bbox[2], table_xs)
	below_boxes = get_below_boxes(bbox, strict = strict)
	below_ids = [x[0] for x in below_boxes]
	below_texts = [get_text_by_id(x) for x in below_ids]
	# print(below_texts)
	# print("+++++++++++++++++")
	return list(zip(below_ids, below_texts))

def get_line_id_by_words(words):
	all_lines=root.find_class('ocr_line')
	text_lines=[get_text_by_line_id(line.get('id')) for line in all_lines if line.get('id') is not None]
	all_lines=[line for line in all_lines if line.get('id') is not None]
	def count_words(text):
		return sum([1 for word in words if str(word).lower() in str(text).lower()])
	all_word_counts=[count_words(x) for x in text_lines]
	max_word_count=max(all_word_counts)
	if max_word_count < 2:
		print('colnames line not found')
		return('colnames line not found')
	rel_indices=[i for i,x in enumerate(all_word_counts) if x==max_word_count]
	rel_index=rel_indices[0]
	rel_line=all_lines[rel_index]
	return rel_line.get('id')

def get_text_by_id(id):
	return str(root.get_element_by_id(id).text.encode('utf-8'))

# get_adj_by_text('Invoice','right')
# get_adj_by_text('Invoice','below')

# get_adj_by_text('Date','right')
# get_adj_by_text('Date','below')

# get_adj_by_text('GST','right')
# get_adj_by_text('GST','below')

def clean_text(text):
	if text:
		text = str(text.strip())
		text = text.replace('|','')
		text = text.replace('[','')
		text = text.replace(']','')
		text = text.replace('(','')
		text = text.replace(')','')
		text = text.strip()
	return text

def check_if_any_words(line, words):
	line = line.lower()
	for word in words:
		if word.strip().lower() in line:
			# if testing:
			# 	print(word, line)
			return True
	return False

def get_seller_by_lines(line_texts):
	alphas = [sum(letter.isalpha() for letter in line) for line in line_texts]
	digits = [sum(letter.isdigit() for letter in line) for line in line_texts]
	line_texts = [line for i,line in enumerate(line_texts) if len(line)>3 and digits[i]<2 and alphas[i]>3]
	line_texts = [line for i,line in enumerate(line_texts) if not check_if_any_words(line, ['Tax','Date','Order','Bank','Invoice', 'Recipient'])]
	# print(line_texts)
	return line_texts[0]

def find_serial_number_id_by_word_id(table_header, table_xs):
	cols = table_header.getchildren()
	serial_number_id = -1
	for i in range(1,len(cols)):
		if cols[i-1].text is not None and 'Des' in cols[i].text:
			serial_number_id = i-1
			# print(cols[i-1].text, cols[i].text, serial_number_id)
			# print(cols[serial_number_id].get('id'))
			break
	if serial_number_id == -1:
		print('Serial Number row not found')
	return cols[serial_number_id].get('id')

def get_enclosing_xrange(x1, x2, table_xs):

	x_coords = [int(x[0]) for x in table_xs]
	x1 = max([x for x in x_coords if x <= x1], default = x1)
	x2 = min([x for x in x_coords if x >= x2], default = x2)

	return (x1, x2)


def get_row_ranges(rownums):

	row_id = line_id
	allowed_ids = {}
	for i in range(len(rownums)):
		try:
			row_id = get_next_id(row_id)
			root_elem = root.get_element_by_id(row_id)
		except KeyError:
			continue
		row_children_id = [x.get('id') for x in root_elem.getchildren()]
		row_text = get_text_by_line_id(row_id)
		# print(row_text)
		alnum = sum([x.isalnum() for x in row_text])
		# print(alnum)
		line_end = sum([True for x in ['cgst', 'sgst','total','output'] if x in row_text])
		line_end = line_end > 0
		if alnum > 4:
			if line_end:
				break
			else:
				allowed_ids[row_id] = row_children_id
	
	# line_ids = [(box_id, row_id_2) for box_id in allowed_ids[row_id_2] for row_id_2 in allowed_ids.keys()]
	all_serial_ids = [x[0] for x in rownums]
	line_ids = []
	for box_id in all_serial_ids:
		for row_id in allowed_ids.keys():
			if box_id in allowed_ids[row_id]:
				elem = (box_id, row_id)
				line_ids.append(elem)
	# print(line_ids)
	# line_ids = [(box_id, row_id_2) for row_id_2 in allowed_ids.keys() for box_id in allowed_ids[row_id_2]]
	# #print(line_ids)
	box_ids = [x[0] for x in line_ids]
	row_bboxes = [get_bbox_by_word_id(x) for x in box_ids]
	row_y1s = [x[1] for x in row_bboxes]
	row_y2s = [x[3] for x in row_bboxes]
	row_ids = [x[1] for x in line_ids]
	row_ranges = list(zip(row_y1s, row_y2s))

	row_ranges = list(zip(box_ids, row_ranges, row_ids))
	return row_ranges

def clean_colname(text):
	text = [x for x in text if x not in [' ','|']]
	text = ''.join(text)
	return text
def get_table_ranges(table_header, table_xs):
	col_ranges= {}
	row_ranges = {}
	col_ids = []
	issues = None
	for col in table_header.getchildren():
		if col.text:
			info = col.get('title')
			bbox = get_bbox_by_title(info)
			x1 = bbox[0]
			x2 = bbox[2]
			col_ids.append(col.get('id'))
			colname = clean_colname(col.text)
			col_ranges[colname] = get_enclosing_xrange(x1, x2, table_xs)
			try:
				serial_word_id = find_serial_number_id_by_word_id(table_header, table_xs)
			except:
				serial_word_id = -1
			if col.get('id') == serial_word_id:
				####### Try strict False here too
				rownums = get_column_by_id(col.get('id'), strict = False)
				row_ranges = get_row_ranges(rownums)				
	if len(row_ranges)==0:
		issues = 'rows not found'
	return row_ranges, col_ranges, col_ids, issues

def get_line_id_by_word_id(word_id):
	root.get_element_by_id

def hocr_to_cv2(p):
	
	x,y=p
	x=x*x_ratio
	y=y*y_ratio

	x=int(x)
	y=int(y)
	p=(x,y)
	return p


def get_table(row_ranges, col_ranges, image, iterative_update_xs=True):
	table = multi_key_dict()
	if iterative_update_xs:
		y1_curr = []
		y2_curr = []
	for i, (colname, dx) in enumerate(col_ranges.items()):
		for j, (snum, dy) in enumerate(row_ranges.items()):
			x1 = dx[0]
			x2 = dx[1]
			y1 = dy[0]
			y2 = dy[1]
			if iterative_update_xs:
				if i>0:
					y1 = y1_curr[j]
					y2 = y2_curr[j]
				else:
					y1_curr.append(y1_curr)
					y2_curr.append(y2_curr)
			# image=cv2.rectangle(image, p1, p2 ,(255,0,0), 5)
			bbox_out = [x1, y1, x2, y2]
			p1 = (x1, y1)
			p2 = (x2, y2)
			p1 = hocr_to_cv2(p1)
			p2 = hocr_to_cv2(p2)
			image=cv2.rectangle(image, p1, p2 ,(255,0,0), 5)
			# out = get_below_boxes(bbox_out, strict= True)
			# #print([root.get_element_by_id(o[0]).text for o in out])
			# image=cv2.rectangle(image, p1, p2 ,(255,0,0), 5)
			rel_bboxes = [(word_ids[i],bbox) for i,bbox in enumerate(word_bboxes) if bbox_within(bbox, bbox_out )]
			# for (id, bbox) in rel_bboxes:
			# 	p1 = (bbox[0], bbox[1])
			# 	p2 = (bbox[2], bbox[3])
			# 	p1 = hocr_to_cv2(p1)
			# 	p2 = hocr_to_cv2(p2)
			# 	image=cv2.rectangle(image, p1, p2 ,(255,0,0), 5)
			if rel_bboxes:
				if iterative_update_xs:
					new_y1 = [x[1][1]for x in rel_bboxes]
					new_y2 = [x[1][3]for x in rel_bboxes]
					new_y1 = sum(new_y1)/len(new_y1)
					new_y2 = sum(new_y2)/len(new_y2)
					new_y1, new_y2 = int(new_y1), int(new_y2)
					y1_curr[j] = new_y1
					y2_curr[j] = new_y2
				# if testing:
				# 	print(snum, colname, [get_text_by_id(x[0]) for x in rel_bboxes])

			else:
				if iterative_update_xs:
					y1_curr[j] = y1
					y2_curr[j] = y2
				# if testing:
					#print(snum, colname, 'No rel_boxes')

		# show_wait_destroy(image)



def get_next_id(line_id):
	line_id = line_id.split('_')
	line_id[-1] = str((int(line_id[-1])+1))
	line_id = '_'.join(line_id)
	return line_id


def fix_colnames(col_ranges):
	for i,col in enumerate(col_ranges):
		if col=='Amoun':
			col_ranges['Amount'] = col_ranges.pop(col)
	col_ranges
	return col_ranges
			
def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def convert_to_amount(text):
	text = clean_text(text)
	num = None
	try:
		num = locale.atoi(text)
	except:
		try:
			num = locale.atof(text)
		except:
			return None
	return num
	

def most_common(lst):
    return max(set(lst), key=lst.count)		

def extract_to_numeric(totals):
	try:
		#print(totals)
		totals = [convert_to_amount(x) for x in totals]
		totals = [t for t in totals if t is not None]
		#print(totals)
		# grand_total = max(totals)
		# totals.remove(grand_total)
		# if int(min(totals)) == 0:
		# 	rounding = min(totals)
		# 	totals.remove(rounding)
		# else:
		# 	rounding = 0.0
		# sub_total = max(totals)
		# totals.remove(sub_total)
		# taxes = [x if i>=rownums for i,x in enumerate(totals)]
		totals.reverse()
		grand_total = max(totals)
		totals = totals[totals.index(grand_total)+1:]

		sub_total = max(totals)
		totals = totals[:totals.index(sub_total)]
		
		rounding = min(totals)
		if int(rounding)==0:
			totals.remove(rounding)

		taxes = {}
		# if len(taxes)>2:
		# 	taxes['GST'] = max(totals)
		# 	texes['CGST'] = most_common(totals)
		# 	taxes['SGST'] = most_common(totals)
		if len(totals)==2 and len(set(totals))==1:
			taxes['CGST'] = totals[0]
			taxes['SGST'] = totals[1]
			taxes['IGST'] = 0
			taxes['Total'] = totals[0]+totals[1]
		elif len(totals)==1:
			taxes['CGST'] = 0
			taxes['SGST'] = 0
			taxes['IGST'] = totals[0]
			taxes['Total'] = totals[0]
		else:
			taxes['CGST'] = 0
			taxes['SGST'] = 0
			taxes['IGST'] = 0
			taxes['Total'] = sum(totals)
		
		output = {'Grand Total': grand_total, "Sub Total": sub_total, "Rounding": rounding, "Taxes": taxes}
		return output
	except:
		#print('totals didnt match up')
		return totals


def get_result_by_name(text):
	out = get_adj_by_text(text)
	# out = sum(out, [])
	return out


def extract_from_row(row, rownum, column_names):

	boxes = {}
	if rownum:
		pass

	hsn_code_flag = False

	def set(new_value, row_i, col_i, append = False):
		if append:
			original_value = boxes[row[row_i]]
			boxes[column_names[col_i]] = ' '.join([original_value, new_value])
		else:
			boxes[column_names[col_i]] = new_value
		
	row_i = 0
	col_i = 0
	while row_i < len(row) and col_i < len(column_names):
		colname = column_names[col_i]
		col_alnum = sum([x.isalnum() for x in colname])
		if col_alnum == 0:
			col_i += 1
			continue
		# #print(row, row_i, col_i, column_names)
		# Extract Sn.
		if col_i==0:
			new_value = row[0]
			if len(row[0])<3 :
				try:
					new_value = locale.atoi(new_value)
				except:
					pass
				set(new_value, row_i, col_i)
			col_i += 1
			row_i += 1
		elif col_i in [1,2,3, 4, 5] and 'Des' in column_names[col_i]:
			description = []
			while row_i < len(row):
				new_value = row[row_i]
				new_value_int = None
				try:
					new_value_int = locale.atoi(new_value)
				except:
					pass
				if ',' not in new_value and new_value_int is not None:
					if len(str(new_value_int))==4 or len(str(new_value_int))==8 :
						hsn_code_flag = True
						break
					else:
						description.append(new_value)
						row_i += 1
				else:
					description.append(new_value)
					row_i += 1
			description_string = ' '.join(description)
			set(description_string, row_i, col_i)
			col_i += 1
		elif col_i in [2,3,4] and 'of' in column_names[col_i].lower():
			col_i += 1
			continue
		elif col_i in [3,4,5] and 'good' in column_names[col_i].lower():
			col_i += 1
			continue
		elif col_i > 1:
				#convert to float
			# 	x=1	
			# 	elif 'rate':
			# 		x = 1
			# 		elif 'unit' in  or 'per' in column_names[col_i].lower():
			# 			x=1
					
			new_value = row[row_i]
			alnum = sum([1 for x in new_value if x.isalnum()])
			if alnum > 0:
				try:
					new_value = locale.atoi(new_value)
				except:
					pass
				try:
					new_value = convert_to_amount(new_value)
				except:
					pass
				set(new_value, row_i, col_i)
				row_i += 1
				col_i += 1
			else:
				row_i += 1
		else:
			col_i += 1
			
	
	amt_value = row[len(row)-1]
	try:
		amt_value = locale.atof(amt_value)
	except:
		try:
			amt_value = locale.atof(row[len(row)-2])
		except:
			pass
		
	set(amt_value, len(row)-1, len(column_names)-1)

	return boxes, hsn_code_flag

def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [ (key + '.' + k, v) for k, v in flatten_dict(value).items() ]
        else:
            return [ (key, value) ]

    items = [ item for k, v in d.items() for item in expand(k, v) ]

    return dict(items)

def add_paths(path1, path2):
	print(path1,path2)
	if type(path1) == type('str'):
		path1 = Path(path1)
	if type(path2) == type('str'):
		path2 = Path(path2)
	path = path1 / path2
	if os_for_pathlib == 'Windows':
		path = PureWindowsPath(path)
	print(path)
	return path
	

if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description=('extract the text within all the ocr_line elements '
					'within the hOCR file')
	)
	parser.add_argument('name', nargs='?', default=sys.stdin)
	args = parser.parse_args()

	jpg_path = add_paths(jpg_folder,  args.name +'.jpg')
	hocr_path = add_paths(hocr_folder, args.name+'.hocr')
	# hocr_path = add_paths(hocr_folder, args.name)#+'.hocr')
	print("hocr_path")
	print(hocr_path)
	doc = html.parse(str(hocr_path))

	#get root
	root = doc.getroot()

	#find all ocr_lines
	ocr_lines = doc.getroot().find_class('ocr_line')

	print('imagecopy_path')
	print(jpg_path)

	# Used for visualization only
	imagecopy = cv2.imread(str(jpg_path))
	h, w, _ = imagecopy.shape 
	w_old=10131
	h_old=14506
	x_ratio = w/w_old
	y_ratio = h/h_old


	word_spans = root.find_class('ocrx_word')
	word_ids = [get_id_by_span(word_span) for word_span in word_spans]
	word_bboxes = [get_bbox_by_word_id(word_id) for word_id in word_ids]


	line_id=get_line_id_by_words(['HSN','Description','Amount','Quantity','Rate','Amount','per','Balance'])
	bbox=get_bbox_by_line_id(line_id)

	# Get the bounding horizontal lines for the table header
	table_xs = get_xs( str(jpg_path) , bbox)
	table_header = root.get_element_by_id(line_id)

	# get the rows according to the end of table marker which gives total amount or total tax
	row_ranges, col_ranges, col_ids, _ = get_table_ranges(table_header, table_xs)
	col_ranges = fix_colnames(col_ranges)

	# Extract all the lines using the row_ids
	next_id = line_id
	table_using_lines = []
	row_ids = [x[2] for x in row_ranges]
	row_ids = unique(row_ids)
	for row_id in row_ids:
		out = get_text_by_line_id(row_id)
		# import pdb; pdb.set_trace()
		out = out.split(' ')
		out = [clean_text(o) for o in out if o]
		#print(out)
		# valid = sum([(x.replace(',','').replace('.','').isdigit()) for x in out if filter_by_expected_value(x, 'Amount')])
		# #print(out, valid)
		# valid = valid > 3
		valid = True
		valid = True
		if valid:
			table_using_lines.append(out)
	#print('Descriptions found, possibly taxes',table_using_lines)


	ymin = min([bbox_xrange[0] for box_id, bbox_xrange, row_id in row_ranges], default = 600 )
	ymax = max([bbox_xrange[1] for box_id, bbox_xrange, row_id in row_ranges], default=99999)

	#print('Rows found',row_ranges)
	# for i,row in row_ranges.items():
	# 	if row[1]-ymax < 500:
	# 		ymax = row[1]+200
	# 	else:
	# 		extra_row.append(i)
	# for i in extra_row:
	# 	row_ranges.pop(i)


	# #print('Rows filtered',row_ranges)

	result = {}

	for i, (colname, row) in enumerate(col_ranges.items()):
		result[colname] = get_adj_by_text(colname, 'below', ymax=ymax, id = col_ids[i])

	table_using_ranges = []
	for box_id, bbox_xrange, row_id in row_ranges:
		row = bbox_xrange
		i = box_id
		for j,col in col_ranges.items():
			if j == "Description":
				x1, x2 = col
				y1, y2 = row
				bbox_out = [x1, y1, x2, y2]
				# p1 = (x1, y1)
				# p2 = (2, y2)
				# p1 = hocr_to_cv2(p1)
				# p2 = hocr_to_cv2(p2)
				rel_bboxes = [(word_ids[i],bbox) for i,bbox in enumerate(word_bboxes) if bbox_within(bbox, bbox_out )]
				# if testing:
				# 	#print(i, j, [get_text_by_id(x[0]) for x in rel_bboxes])
				iou_values = [(word_ids[i], bb_intersection_over_union(bbox,bbox_out)) for i,bbox in enumerate(word_bboxes)]
				iou_values = sorted(iou_values, key = lambda x: x[1], reverse= True)
				line = ([get_text_by_id(x[0]) for x in  iou_values[:3]])
				table_using_ranges.append(' '.join([clean_text(x) for x in line if x]))
				# imagecopy = cv2.rectangle(imagecopy, p1, p2 ,(255,0,0), 5)
	# show_wait_destroy(imagecopy)

	# Extract totals by using all the amounts below the 'Amount' column
	# If 'Amount' isn't in col_ranges use the last column
	try:
		amount_xrange = col_ranges['Amount']
	except:
		x1 = max([v[0] for k, v in col_ranges.items()])
		x2 = max([v[1] for k, v in col_ranges.items()])
		amount_xrange = (x1,x2)
	
	amount_xrange = get_enclosing_xrange(amount_xrange[0], amount_xrange[1], table_xs)
	amount_bbox = amount_xrange[0], ymin, amount_xrange[1], ymin
	totals = get_below_boxes(amount_bbox,strict = False, ymax=99999, text='Amount', extract='Totals')
	if len(totals)<=2:
		#print('expanding amount box')
		amount_bbox_2 = amount_xrange[0]-200 , ymin, amount_xrange[1]+200 , ymax
		totals = get_below_boxes(amount_bbox_2,strict = False, ymax=99999, text='Amount', extract='Totals')
	totals = [get_text_by_id(x[0]) for x in totals]
	totals = filter_by_expected_value(totals, 'Amount')
	totals = extract_to_numeric(totals)

	# get_table(row_ranges, col_ranges, imagecopy)

	# clean by number or letter - each row
	# removw variable length descriptions
	# categorize header and rows into datatypes ex 
	# -> serial no 
	# -> int i.e. nos or hsn or sac 
	# -> percentage i.e. % sign
	# -> 


	# rates = []
	# for i,row in row_ranges.items():
	# 	for j,col in col_ranges.items():
	# 		if j == 'Rate':
	# 			x1, x2 = col
	# 			y1, y2 = row
	# 			bbox_out = [x1, y1, x2, y2]
	# 			rel_bboxes = [(word_ids[i],bbox) for i,bbox in enumerate(word_bboxes) if bbox_within(bbox, bbox_out )]
	# 			#print(i, j, [get_text_by_id(x[0]) for x in rel_bboxes])
	# 			iou_values = [(word_ids[i], bb_intersection_over_union(bbox,bbox_out)) for i,bbox in enumerate(word_bboxes)]
	# 			iou_values = sorted(iou_values, key = lambda x: x[1], reverse= True)
	# 			rate = ([get_text_by_id(x[0]) for x in  iou_values[0:1]])
	# 			rates.append(rate)
	# #print(rates)

	date = get_result_by_name('Date')
	if len(date)>0:
		date = date[0]
	#print(">",date)
	# pdb.set_trace()
	line_texts = [get_text_by_line_id(x.get('id')) for x in ocr_lines if x.get('id')]
	all_words = ' '.join(line_texts)
	gst = re.findall("\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}",all_words)
	if len(gst)>0:
		seller_gst = gst[0]
	else:
		seller_gst = 'Not Found'
	if len(gst)>1:
		buyer_gst = gst[1]
	else:
		buyer_gst = 'Not found'
	#print(">",gst)
	invoice = get_result_by_name('Invoice')
	if len(invoice)>0:
		invoice = invoice[0]
	#print(">",invoice)

	gst_line = [i for i, line in enumerate(line_texts) if 'GST'.lower() in line.lower()]
	if gst_line:
		gst_line = gst_line[0]
		ocr_lines_till_gst=ocr_lines[0:gst_line]

	gst_line = [i for i, line in enumerate(line_texts) if 'State'.lower() in line.lower()]
	if gst_line:
		gst_line = gst_line[0]
		ocr_lines_till_gst=ocr_lines[0:gst_line]

	line_fsizes = [get_line_fsize(line) for line in ocr_lines_till_gst]
	line_fsizes = sorted(line_fsizes, key=lambda x: x[1], reverse = True)
	line_texts = [get_text_by_line_id(x[0]) for x in line_fsizes if x[0]]

	if line_texts:
		seller = get_seller_by_lines(line_texts)
	else:
		seller = 'Not Found'
	#print(">",seller)

	# Get Buyer / Currently using 'BALWANT INDUSTRIES'

	# buyer = get_result_by_name('Buyer')
	# buyer.append(get_result_by_name('Consignee'))
	buyer = 'BALWANT INDUSTRIES'
	#print(">",buyer)
	output = {"Seller Name": seller, "Buyer Name": buyer, "Invoice Number": invoice,
		'Date': date, 'Seller GST': seller_gst, 'Buyer GST': buyer_gst,  "Invoice Level": totals}

	column_names_original = [x.text if x.text is not None else 'None' for x in table_header.iterchildren()]
	column_names = [clean_text(x.text) for x in table_header.iterchildren() if x.text is not None]

	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FINAL OUTPUT <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

	print("Metadata from Receipt")
	print("\n")
	for x in output.items():
		print(x[0], ":", x[1])
	print("\n")
	print("\n")

	# output["Description Table"]=result

	print("Lines extracted from OCR")
	print("\n")
	table_with_headers = pd.DataFrame(columns=column_names)

	for rownum, row in enumerate(table_using_lines):
		extracted_row_dict, hsn_code_flag = extract_from_row(row, rownum, column_names)
		if hsn_code_flag:
			print("\n")
			print("Extracted line from table: ", row)
			print("Parsed into individual elements by vertical align:", extracted_row_dict)
			table_with_headers = table_with_headers.append(extracted_row_dict, ignore_index=True)
	print("\n")
	print("\n")

	print("Lines parsed to output")
	print("\n")
	print(table_with_headers)
	print("\n")


	output_flat = flatten_dict(output)
	output_flat = pd.Series(output_flat)
	output_pages = [output_flat, table_with_headers]
	output_path = Path("invoice2") / Path("output") / Path("{}.xlsx".format(str(args.name)))
	print("NOTE: Please find the xlsx sheet in the folder: {}".format(str(output_path)) )
	print("\n")
	print("\n")
	print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> - <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
	writer = pd.ExcelWriter( output_path)
	for i, df in enumerate(output_pages):
		df.to_excel(writer,'sheet{}'.format(i))
	writer.save() 

